# ADR 0001: Event-driven clean-room architecture

Status: accepted

We use immutable domain events feeding a deterministic decision/execution path. Research components exchange typed records and have no order authority. This makes reconstruction and leakage testing straightforward and avoids conversational multi-agent trade decisions. The implementation was designed from this project's stated constraints; no external trading-system source or orchestration was consulted.

