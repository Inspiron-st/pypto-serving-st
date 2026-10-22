# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Growing and shrinking the set of replicas the router owns.

The pool decides *where* a replica can go; this decides *when* one appears and
disappears, and keeps the registry in step. It also owns the two asymmetries a
model load forces:

* growing is slow and must not be waited on -- a replica is registered
  unroutable and promoted by the health poller minutes later;
* shrinking must not be abrupt -- a replica stops taking new sessions, finishes
  what it holds, and only then is stopped.

Underneath both runs one invariant: **a device group stays reserved until the
replica that held it is confirmed retired**. Concurrency and restarts are the
two ways to lose it, so retirement is shared per *instance* rather than per slot
name, and every unresolved reservation is written to the state file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pypto_serving.router.config import ReplicaSpec, RouterConfig
from pypto_serving.router.launcher import (
    HostPool,
    LaunchError,
    PoolExhausted,
    Slot,
    transport_for,
)
from pypto_serving.router.routing import ReplicaRegistry

logger = logging.getLogger(__name__)

# How long to wait for a signalled replica to actually stop answering before
# reusing its device anyway. A replica that ignores its stop must not hold a
# card for the life of the router.
STOP_GRACE_SECONDS = 30.0

# Why a slot is reserved with nothing of ours serving on it. Reported through
# capacity() so an operator can see why the fleet is smaller than the config.
STOP_UNCONFIRMED = "stop-unconfirmed"
UNREACHABLE_AT_STARTUP = "unreachable-at-startup"


@dataclass
class _Instance:
    """One launch onto one slot.

    The slot is reused by whatever replica comes next; the instance never is.
    Anything that can outlive a replica -- a retirement still waiting for the old
    process to go quiet -- is keyed on the instance, so a late completion cannot
    act on the replica that took its place.
    """

    slot: Slot
    generation: int
    spec: ReplicaSpec
    # One retirement per instance, shared by every caller that asks to stop it.
    retirement: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return self.slot.name


@dataclass(frozen=True)
class _Reserved:
    """A slot held with nothing of ours serving on it, and why."""

    slot: Slot
    reason: str


class FleetManager:
    """Launches, drains and stops the replicas this router owns."""

    def __init__(
        self,
        config: RouterConfig,
        registry: ReplicaRegistry,
        *,
        transport_factory=transport_for,
        state_path: Path | None = None,
        probe=None,
        alive_probe=None,
        stop_grace: float = STOP_GRACE_SECONDS,
    ) -> None:
        self._config = config
        self._registry = registry
        self._pool = HostPool(config, transport_factory=transport_factory)
        self._state_path = state_path
        # Two questions, deliberately not one probe:
        #   probe        -- "is a replica *serving* here?" (readiness)
        #   alive_probe  -- "is *anything* answering here?" (any HTTP status)
        # Adoption asks the first, confirming an exit asks the second. Answering
        # the second with the first frees the devices of a process that is still
        # holding them; see _wait_gone.
        self._probe = probe
        self._alive_probe = alive_probe
        self._stop_grace = stop_grace
        # name -> instance, for every replica this router started.
        self._owned: dict[str, _Instance] = {}
        self._watchers: dict[str, asyncio.Task] = {}
        # Slots reserved with nothing of ours on them: a stop that could not be
        # confirmed, or a replica a previous run recorded that did not answer at
        # startup. They stay taken in the pool, so nothing is launched onto
        # devices that may still be in use, and they are reported so an operator
        # can see why the capacity is lower than the configuration says.
        self._unconfirmed: dict[str, _Reserved] = {}
        self._generation = 0

    @property
    def pool(self) -> HostPool:
        return self._pool

    @property
    def owned_names(self) -> tuple[str, ...]:
        return tuple(self._owned)

    def capacity(self) -> dict:
        return {
            "owned": len(self._owned),
            "ceiling": self._pool.ceiling,
            "free_slots": [slot.name for slot in self._pool.free_slots()],
            # Reserved but not serving, and why.
            "unconfirmed": {
                name: held.reason for name, held in sorted(self._unconfirmed.items())
            },
        }

    # --- growing ----------------------------------------------------------

    async def launch_one(self) -> ReplicaSpec:
        """Reserve a slot, start a replica on it, and register it unroutable.

        Returns as soon as the launch command has been accepted. The replica is
        not usable yet; the health poller promotes it when its model is up.
        """
        slot = self._pool.allocate()  # synchronous: no await may split this
        try:
            launched = await self._pool.launch(slot)
        except LaunchError:
            # launch() already released the slot.
            raise
        self._adopt_instance(slot, launched.spec, ready=False)
        self._write_state()
        return launched.spec

    def _adopt_instance(self, slot: Slot, spec: ReplicaSpec, *, ready: bool) -> _Instance:
        """Record a replica on an already-reserved slot, and watch it come up."""
        self._generation += 1
        instance = _Instance(slot=slot, generation=self._generation, spec=spec)
        self._owned[instance.name] = instance
        self._registry.add(spec, ready=ready, owned=True)
        if not ready:
            self._watchers[instance.name] = asyncio.create_task(
                self._watch_launch(instance.name)
            )
        return instance

    async def _watch_launch(self, name: str) -> None:
        """Stop a replica that never becomes ready, so its device comes back."""
        deadline = self._config.launch_timeout_seconds
        waited = 0.0
        # Never slower than 5s (a model load is minutes), never faster than 50ms,
        # and never coarser than the deadline itself, so a short timeout is
        # honoured rather than rounded up to the next poll.
        step = max(0.05, min(5.0, self._config.health_interval_seconds, deadline))
        try:
            while waited < deadline:
                state = self._registry.state(name)
                if state is None:
                    return  # removed by someone else
                if state.ready:
                    logger.info("replica %s reported ready after %.0fs", name, waited)
                    return
                await asyncio.sleep(step)
                waited += step
        except asyncio.CancelledError:
            raise
        logger.error(
            "replica %s never became ready within the %.0fs launch timeout; stopping it and "
            "freeing its device(s). If the model was still loading -- DeepSeek V4 needs well "
            "over 1200s from cold -- raise --launch-timeout; %s holds the startup output",
            name, deadline, self._pool.log_path_for(name) or "the host's log_dir",
        )
        await self.stop(name, drain=False)

    async def launch_initial(self) -> list[ReplicaSpec]:
        """Start the replicas the configuration asks for at boot.

        Concurrent, because N model loads in parallel cost one load's wall clock
        rather than N of them.
        """
        # Replicas adopted from a previous run already count towards the request.
        # Without this the fleet grows by initial_replicas on *every* restart
        # until it hits the ceiling -- a card per restart on a shared cluster,
        # each one a fresh model load -- and then reports a spurious
        # "initial launch failed: PoolExhausted" for a perfectly healthy adoption.
        wanted = self._config.initial_replicas - len(self._owned)
        if wanted <= 0:
            return []
        results = await asyncio.gather(
            *(self.launch_one() for _ in range(wanted)), return_exceptions=True
        )
        launched: list[ReplicaSpec] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error("initial launch failed: %s", result)
            else:
                launched.append(result)
        return launched

    # --- shrinking --------------------------------------------------------

    async def stop(self, name: str, *, drain: bool = True) -> bool:
        """Drain (optionally) and stop a replica this router owns.

        Every caller asking to stop the same instance shares one retirement.
        Running one each is what lost slots: two concurrent DELETEs both passed
        the ownership check, the second found the registry entry already gone,
        read that as proof of exit, and released the slot while the first was
        still waiting for the endpoint to go quiet. If a replacement had been
        launched onto that slot in between, the first one's cleanup then retired
        *it* -- leaving a live replica nobody owned on devices the pool was free
        to hand out again.
        """
        instance = self._owned.get(name)
        if instance is None:
            return False
        self._cancel_watcher(name)
        if instance.retirement is None or instance.retirement.cancelled():
            instance.retirement = asyncio.create_task(self._retire(instance, drain=drain))
        # Shielded: a client that disconnects mid-DELETE cancels this coroutine,
        # and a retirement abandoned after the stop was signalled would leave a
        # dead replica owning its slot with nothing left to release it.
        return await asyncio.shield(instance.retirement)

    def _cancel_watcher(self, name: str) -> None:
        """Drop the launch watcher of a replica that is going away.

        Never cancelled from inside itself: the watcher calls stop() when a
        launch overruns, so cancelling it there would deliver the CancelledError
        to the very coroutine asking for the retirement.
        """
        watcher = self._watchers.pop(name, None)
        if watcher is None or watcher.done() or watcher is asyncio.current_task():
            return
        watcher.cancel()

    async def _retire(self, instance: _Instance, *, drain: bool) -> bool:
        """Take one instance out of rotation and stop it. One per instance."""
        name = instance.name
        try:
            if drain:
                await self._drain(name)
            state = self._registry.remove(name)
            spec = state.spec if state is not None else instance.spec
            signalled = await self._pool.signal_stop(instance.slot)
            # Only now is the device really free. Releasing the slot on the
            # signal alone would let the next launch bind a port the old replica
            # still holds, and that failure would look like a bad configuration.
            gone = await self._wait_gone(spec)
        except Exception:  # noqa: BLE001 - whatever failed, the devices are unknown
            self._registry.remove(name)
            logger.exception("retiring replica %s failed", name)
            signalled = gone = False
        return self._finish_retirement(instance, signalled=signalled, gone=gone)

    def _finish_retirement(self, instance: _Instance, *, signalled: bool, gone: bool) -> bool:
        name = instance.name
        if self._owned.get(name) is not instance:
            # Checked rather than assumed. The slot stays taken in the pool for
            # the whole retirement, so nothing should be able to take it in the
            # meantime -- but acting on a replacement is precisely the failure
            # this instance identity exists to make impossible, so it is worth a
            # branch instead of a comment.
            logger.warning(
                "the retirement of %s (generation %d) finished after its slot changed hands; "
                "leaving the current owner alone",
                name, instance.generation,
            )
            return signalled and gone

        self._owned.pop(name, None)
        if not (signalled and gone):
            # The replica is out of rotation either way, but its devices are in
            # an unknown state: the stop command failed, or something is still
            # answering. Reusing them could overlap two instances on the same
            # cards, so the slot stays reserved and the retirement is reported
            # as incomplete. Freeing it is a human decision for now.
            self._unconfirmed[name] = _Reserved(instance.slot, STOP_UNCONFIRMED)
            self._write_state()
            logger.error(
                "replica %s could not be confirmed stopped (signalled=%s, gone=%s); "
                "slot %s stays reserved",
                name, signalled, gone, instance.slot.name,
            )
            return False

        self._pool.release(name)
        self._write_state()
        logger.info("replica %s stopped; slot %s is free", name, instance.slot.name)
        return True

    async def _wait_gone(self, spec: ReplicaSpec, timeout: float | None = None) -> bool:
        """Wait until nothing answers at a stopped replica's address.

        The question here is liveness, not readiness: *any* HTTP answer means
        something is still there, 503 included. serving answers /health with 503
        whenever its engine is not ready, so taking a 503 for an exit would
        release the devices of a process that is still holding them -- and a
        worker that died while its HTTP server stayed up is exactly when that
        answer appears.

        A silent endpoint is still weak evidence -- it says the HTTP server is
        gone, not that the serving process and its device workers have exited --
        but a noisy one is conclusive: something is there, so the devices are
        not free.

        TODO: a stronger check needs a verifiable identity for what was launched
        (a pid, or the broker task id where there is a broker) instead of an
        address. Worth a configurable per-host probe, like ``stop_command``, if
        the endpoint test proves too weak in practice.
        """
        # The readiness probe is the weaker answer to this question -- it calls a
        # 503 "gone" -- so it is a fallback for callers that supply only one, not
        # the default the router runs with.
        probe = self._alive_probe or self._probe
        if probe is None:
            return True
        timeout = self._stop_grace if timeout is None else timeout
        waited = 0.0
        while waited < timeout:
            if not await probe(spec):
                return True
            await asyncio.sleep(0.5)
            waited += 0.5
        logger.error(
            "%s still answered %s after %.0fs; keeping its slot reserved",
            spec.name, spec.base_url, timeout,
        )
        return False

    async def _drain(self, name: str) -> None:
        """Stop routing to a replica and wait for its in-flight work.

        Bounded: a stuck request must not hold a device forever, but the bound
        is generous because cutting a user's generation short to reclaim a card
        is the worse trade.
        """
        self._registry.set_draining(name, True)
        deadline = self._config.drain_timeout_seconds
        waited = 0.0
        while waited < deadline:
            state = self._registry.state(name)
            if state is None or state.outstanding <= 0:
                return
            await asyncio.sleep(0.25)
            waited += 0.25
        state = self._registry.state(name)
        remaining = state.outstanding if state else 0
        logger.warning(
            "replica %s still had %d request(s) in flight after %.0fs; stopping anyway",
            name, remaining, deadline,
        )

    async def stop_all(self) -> None:
        """Stop every owned replica. Called on an explicit shutdown, not a crash."""
        names = list(self._owned)
        if not names:
            return
        logger.info("stopping %d replica(s) this router launched", len(names))
        await asyncio.gather(*(self.stop(name) for name in names), return_exceptions=True)

    # --- crash recovery ---------------------------------------------------

    @staticmethod
    def _state_entry(slot: Slot, *, state: str, reason: str | None = None) -> dict:
        entry = {
            "slot": slot.name,
            "host": slot.host.name,
            "devices": list(slot.devices),
            "address": slot.host.address(),
            "port": slot.port,
            "state": state,
        }
        if reason is not None:
            entry["reason"] = reason
        return entry

    def _write_state(self) -> None:
        if self._state_path is None:
            return
        # Unresolved reservations are written too. Recording only what is owned
        # dropped a slot from the file at exactly the moment it mattered -- a
        # stop that could not be confirmed -- so the next run saw that device
        # group as free and could launch onto cards the old process may still
        # have held.
        payload = {
            "replicas": [
                self._state_entry(instance.slot, state="owned")
                for instance in self._owned.values()
            ] + [
                self._state_entry(held.slot, state="unconfirmed", reason=held.reason)
                for held in self._unconfirmed.values()
            ]
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            # Temp file plus os.replace, because write_text truncates before it
            # writes: a crash mid-write would leave a half-written file, and
            # surviving a crash is the only reason this file exists.
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:
            logger.warning("could not write fleet state to %s: %s", self._state_path, exc)

    def _hold(self, slot: Slot, reason: str, detail: str) -> None:
        """Keep a slot reserved with nothing of ours on it, and say why."""
        self._unconfirmed[slot.name] = _Reserved(slot, reason)
        logger.error(
            "slot %s stays reserved (%s): %s. Free it by stopping whatever holds "
            "%s device(s) %s and restarting the router",
            slot.name, reason, detail, slot.host.name, list(slot.devices),
        )

    async def adopt_previous(self, probe=None) -> list[str]:
        """Re-take replicas a previous run of this router left behind.

        A crash runs no shutdown hook, so its replicas keep serving. Throwing
        them away would cost a fresh model load each; probing and re-adopting
        the ones that answer costs a round trip.

        The recorded reservation is restored *before* anything is probed. A
        replica that was still loading its model when the router died answers
        nothing yet -- uvicorn binds only once the engine is up -- and so does a
        healthy one behind a blip in the network. Classifying first and
        reserving only on success let launch_initial() put a second replica on
        those same devices.
        """
        if self._state_path is None or not self._state_path.exists():
            return []
        try:
            recorded = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ignoring unreadable fleet state %s: %s", self._state_path, exc)
            return []

        # Everything below is parsed defensively. The file is operator-visible
        # (--state-file) and may be hand-edited or written by another version, and
        # adopt_previous is awaited from the lifespan with no handler: an
        # AttributeError or ValueError here does not degrade to "no adoption", it
        # stops the router from starting at all.
        if not isinstance(recorded, dict):
            logger.warning("ignoring fleet state %s: not a JSON object", self._state_path)
            return []
        entries = recorded.get("replicas", [])
        if not isinstance(entries, list):
            logger.warning("ignoring fleet state %s: 'replicas' is not a list", self._state_path)
            return []

        adopted: list[str] = []
        for entry in entries:
            name = self._recorded_name(entry)
            if name is None:
                continue
            slot = self._pool.reserve(name)
            if slot is None:
                logger.warning(
                    "slot %s is no longer declared in the config, or is already taken; "
                    "not adopting", name,
                )
                continue
            if entry.get("state") == "unconfirmed":
                # The previous run could not confirm this one stopped. Nothing
                # since then has, either. The recorded reason is matched against
                # the known ones rather than echoed: the file is hand-editable
                # and capacity() is an API response.
                reason = entry.get("reason")
                self._hold(
                    slot,
                    reason if reason in (STOP_UNCONFIRMED, UNREACHABLE_AT_STARTUP)
                    else STOP_UNCONFIRMED,
                    "the previous run could not confirm it stopped",
                )
                continue
            if await self._adopt_recorded(slot, entry, probe):
                adopted.append(name)

        if adopted:
            logger.info("adopted %d replica(s) from the previous run: %s",
                        len(adopted), ", ".join(adopted))
        self._write_state()
        return adopted

    def _recorded_name(self, entry) -> str | None:
        """The slot name of a state-file entry worth acting on, if it has one."""
        if not isinstance(entry, dict):
            logger.warning("ignoring a malformed entry in %s: %r", self._state_path, entry)
            return None
        name = entry.get("slot")
        if not isinstance(name, str) or not name:
            return None
        if self._registry.state(name) is not None:
            return None
        return name

    async def _adopt_recorded(self, slot: Slot, entry: dict, probe) -> bool:
        """Classify one recorded replica on its restored slot.

        Ready, still starting, or unreachable -- and the slot stays reserved for
        all three. Only the first two are adopted into the registry.
        """
        try:
            port = int(entry.get("port", 0))
        except (TypeError, ValueError):
            port = 0
        address = entry.get("address", "")
        if not isinstance(address, str) or not address or not port:
            self._hold(slot, UNREACHABLE_AT_STARTUP,
                       f"its recorded address is unusable ({address!r}:{entry.get('port')!r})")
            return False

        spec = ReplicaSpec(name=slot.name, host=address, port=port)
        ready = probe if probe is not None else self._probe
        if ready is not None and await ready(spec):
            self._adopt_instance(slot, spec, ready=True)
            return True
        # Not serving. If anything still answers there it is ours and it is
        # coming up, so adopt it unroutable and let the health poller promote it.
        # It is watched like a fresh launch, so one that never comes up is
        # stopped at --launch-timeout and gives its devices back.
        alive = self._alive_probe
        if alive is not None and await alive(spec):
            logger.info(
                "replica %s from the previous run answers at %s but is not ready; "
                "adopting it as still starting",
                slot.name, spec.base_url,
            )
            self._adopt_instance(slot, spec, ready=False)
            return True
        self._hold(slot, UNREACHABLE_AT_STARTUP,
                   f"it did not answer at {spec.base_url}")
        return False


__all__ = ["FleetManager", "PoolExhausted"]
