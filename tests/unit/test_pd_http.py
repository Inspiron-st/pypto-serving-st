# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host HTTP ordering and request normalization for the PD Router."""

import asyncio
import json

from fastapi import HTTPException
import httpx
import msgspec
import pytest
from starlette.requests import Request

from pypto_serving.pd.config import Config, Endpoint
from pypto_serving.pd.protocol import Binding, Component, Contract, Node, Topology, decode, digest
from pypto_serving.pd.router import Router, create_app


def config():
    return Config((Endpoint("p", 8111, "p"),), (Endpoint("d", 8112, "d"),),
                  capacity=1, handoff_timeout=0.05, request_timeout=1, log_enabled=False)


def descriptor(name):
    return Node(name, "prefill" if name == "p" else "decode", "boot-" + name,
                Contract("test", "model", "layout", (Component("kv", "kv", "kv"),)),
                "registry-" + name, Topology.for_tp(1, 1))


def request(payload, path="/v1/completions"):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
    return Request({"type": "http", "method": "POST", "scheme": "http", "server": ("router", 8110),
                    "path": path, "query_string": b"", "headers": []}, receive)


class Body(httpx.AsyncByteStream):
    def __init__(self):
        self.closes = 0

    async def __aiter__(self):
        yield b'{"choices":[]}'

    async def aclose(self):
        self.closes += 1


@pytest.mark.parametrize("failure", ["prefill", "cancel", "timeout"])
def test_cleanup_closes_decode_response_returned_after_abort(failure):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        body, responses, aborts = Body(), [], []
        async def transport(req):
            if req.url.path.endswith("descriptor"):
                return httpx.Response(200, json=msgspec.to_builtins(descriptor(req.url.host)))
            if req.url.path.endswith("abort"):
                aborts.append(req.url.host)
                release.set()
                return httpx.Response(200, json={"accepted": True})
            if req.url.host == "p":
                await started.wait()
                if failure != "prefill":
                    await release.wait()
                return httpx.Response(500, json={"error": "prefill failed"})
            started.set()
            await release.wait()
            response = httpx.Response(200, stream=body)
            responses.append(response)
            return response
        router = Router(config(), httpx.MockTransport(transport))
        try:
            task = asyncio.create_task(router.generate(request({"stream": failure == "timeout"})))
            if failure == "cancel":
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
            error = {"prefill": HTTPException, "cancel": asyncio.CancelledError, "timeout": RuntimeError}[failure]
            with pytest.raises(error):
                await asyncio.wait_for(task, 1)
            assert sorted(aborts) == ["d", "p"]
            assert len(responses) == 1 and responses[0].is_closed and body.closes == 1
            # A single cleanup returned both gates; the next request can acquire them.
            await asyncio.wait_for(router.requests.acquire(), 0.1)
            await asyncio.wait_for(router.handoffs.acquire(), 0.1)
            assert router.requests.locked() and router.handoffs.locked()
        finally:
            for response in responses:
                await response.aclose()
            await router.client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("fields,streaming", [({}, False), ({"stream": False}, False),
    ({"stream": "false"}, False), ({"stream": "0"}, False), ({"stream": "off"}, False),
    ({"stream": True}, True), ({"stream": "true"}, True), ({"stream": 0}, False),
    ({"stream": 1}, True), ({"stream": None}, None), ({"stream": "invalid"}, None)])
def test_stream_coercion_matches_model_node_schema(kind, fields, streaming):
    from pypto_serving.serving.server.server import ChatCompletionRequest, CompletionRequest
    async def run():
        posts = []
        schema = CompletionRequest if kind == "completion" else ChatCompletionRequest
        async def transport(req):
            if req.url.path.endswith("descriptor"):
                return httpx.Response(200, json=msgspec.to_builtins(descriptor(req.url.host)))
            if req.url.path.endswith("abort"):
                return httpx.Response(200, json={"accepted": True})
            body = json.loads(req.content)
            public = schema.model_validate(body)
            binding = decode(json.loads(req.headers["x-pd-context"])["binding"], Binding)
            assert binding.input_digest == digest(body)
            assert type(body["stream"]) is bool and public.stream == streaming
            posts.append(req.url.host)
            if req.url.host == "p":
                return httpx.Response(200, json={"request": binding.request, "status": "ready"})
            if public.stream:
                return httpx.Response(200, text='data: {"choices":[{"text":"answer","finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                                      headers={"content-type": "text/event-stream"})
            # A coerced-false request must use the longer JSON header deadline.
            await asyncio.sleep(config().handoff_timeout * 2)
            return httpx.Response(200, json={"choices": [{"text": "answer"}]})
        app = create_app(config(), httpx.MockTransport(transport))
        payload = {"prompt": "hello"} if kind == "completion" else {
            "messages": [{"role": "user", "content": "hello"}]}
        path = "/v1/completions" if kind == "completion" else "/v1/chat/completions"
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
                response = await asyncio.wait_for(client.post(path, json={**payload, **fields}), 2)
            if streaming is None:
                assert response.status_code == 400 and not posts
            else:
                assert response.status_code == 200 and "answer" in response.text
                assert "pd_execution_error" not in response.text and sorted(posts) == ["d", "p"]
                media = "text/event-stream" if streaming else "application/json"
                assert response.headers["content-type"].startswith(media)
        finally:
            await app.state.router.client.aclose()
    asyncio.run(run())
