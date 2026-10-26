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
from pypto_serving.pd.protocol import Component, Contract, Node, Topology
from pypto_serving.pd.router import Router


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
