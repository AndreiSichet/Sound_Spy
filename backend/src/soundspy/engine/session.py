"""``Session``: one live or offline processing session.

Binds an ``AudioSource``, a ``ProcessorChain`` and the ``AudioSink``(s) to a worker
thread: source callback -> bounded input queue -> worker (runs the chain) -> output queue
-> sink. Side events are handed to the asyncio loop with ``loop.call_soon_threadsafe``.
Target changes are applied atomically at the next chunk boundary.

TODO(step-03): implement.
"""
