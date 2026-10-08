"""Single owner of the native Generator; independent queues per request."""
from __future__ import annotations

import contextlib
import queue
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field

from protocol import APIError


@dataclass
class Ticket:
    kwargs: dict
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=128))
    cancelled: threading.Event = field(default_factory=threading.Event)
    job: object = None


class GenerationWorker:
    def __init__(self, generator, job_factory, max_active=1, max_pending=32,
                 inference_context=contextlib.nullcontext):
        self.generator, self.job_factory = generator, job_factory
        self.max_active = max_active
        self.inference_context = inference_context
        self.inbox = queue.Queue()
        self.capacity = threading.BoundedSemaphore(max_active + max_pending)
        self.lock = threading.Lock()
        self.tickets = {}
        self.failed = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="native-generator", daemon=True)
        self.thread.start()

    def submit(self, **kwargs):
        with self.lock:
            if self.failed or self.closed or not self.thread.is_alive():
                raise APIError("Inference worker unavailable", 503, "engine_unavailable")
            if not self.capacity.acquire(blocking=False):
                raise APIError("Inference queue is full", 429, "queue_full")
            ticket = Ticket(kwargs)
            self.tickets[ticket.identifier] = ticket
            self.inbox.put(ticket)
            return ticket

    def cancel(self, ticket):
        # Generator.cancel itself is called only by the generator owner.
        ticket.cancelled.set()

    def _retire(self, ticket):
        with self.lock:
            if self.tickets.pop(ticket.identifier, None) is not None:
                self.capacity.release()

    def _error(self, ticket, error):
        while True:
            try:
                ticket.events.get_nowait()
            except queue.Empty:
                break
        ticket.events.put_nowait(error)
        self._retire(ticket)

    def _run(self):
        active, waiting = {}, deque()
        try:
            with self.inference_context():
                while not self.closed:
                    if not active and not waiting:
                        try:
                            waiting.append(self.inbox.get(timeout=0.1))
                        except queue.Empty:
                            continue
                    while True:
                        try:
                            waiting.append(self.inbox.get_nowait())
                        except queue.Empty:
                            break
                    for ident, ticket in list(active.items()):
                        if ticket.cancelled.is_set():
                            self.generator.cancel(ticket.job)
                            del active[ident]
                            self._retire(ticket)
                    while waiting and len(active) < self.max_active:
                        ticket = waiting.popleft()
                        if ticket.cancelled.is_set():
                            self._retire(ticket)
                            continue
                        try:
                            ticket.job = self.job_factory(identifier=ticket.identifier, **ticket.kwargs)
                            self.generator.enqueue(ticket.job)
                        except (ValueError, AssertionError) as exc:
                            self._error(ticket, APIError(f"Request cannot be queued: {exc}"))
                            continue
                        active[ticket.identifier] = ticket
                    if not active:
                        continue
                    for event in self.generator.iterate():
                        ticket = active.get(event.get("identifier"))
                        if ticket is None:
                            continue
                        # Avoid retaining GPU tensors, logits, or Job references
                        # in slow HTTP queues.
                        safe = {k: event[k] for k in (
                            "identifier", "stage", "text", "eos", "eos_reason", "new_tokens",
                            "prompt_tokens", "time_enqueued", "time_prefill", "time_generate",
                            "accepted_draft_tokens", "rejected_draft_tokens") if k in event}
                        try:
                            ticket.events.put_nowait(safe)
                        except queue.Full:
                            self.generator.cancel(ticket.job)
                            del active[ticket.identifier]
                            self._error(ticket, APIError("Client is consuming output too slowly", 503, "slow_consumer"))
                            continue
                        if event.get("eos"):
                            del active[ticket.identifier]
                            self._retire(ticket)
                    if active and not self.generator.num_remaining_jobs():
                        raise RuntimeError("Generator drained without terminal events")
        except Exception:
            # Fail closed rather than reloading or continuing an unknown GPU
            # state. Do not include model input or raw exception text in HTTP.
            with self.lock:
                self.failed = "Inference worker failed; inspect the server log"
            import traceback
            traceback.print_exc()
        finally:
            for ticket in list(active.values()):
                try:
                    self.generator.cancel(ticket.job)
                except Exception:
                    pass
            with self.lock:
                outstanding = list(self.tickets.values())
            for ticket in outstanding:
                self._error(ticket, APIError(self.failed or "Server is closing", 503, "engine_unavailable"))

    def status(self):
        with self.lock:
            return {"healthy": not self.failed and not self.closed and self.thread.is_alive(),
                    "requests": len(self.tickets), "max_active_requests": self.max_active}

    def close(self):
        self.closed = True
        self.thread.join(timeout=5)
