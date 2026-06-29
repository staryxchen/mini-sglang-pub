# PD Disaggregation Design — Teaching-Level PoC

## Context

mini-sglang currently runs prefill and decode in the same Scheduler process. The goal is to split them into independent P (Prefill) and D (Decode) workers, connected by a KV cache transfer layer. This enables:
- Independent scaling of prefill vs decode capacity
- Better GPU utilization (prefill is compute-bound, decode is memory-bound)

Scope: PoC to verify the core flow works end-to-end (1P:1D, single request lifecycle).

---

## Current Architecture (reference)

```
Frontend (FastAPI)
  ↕ ZMQ (async push/pull)
Tokenizer/Detokenizer
  ↕ ZMQ (push/pull, msgpack)
Scheduler (per TP rank)
  ├── PrefillManager → schedule prefill batch
  ├── DecodeManager  → schedule decode batch
  ├── CacheManager   → page allocation, prefix cache
  └── Engine         → model forward, KV cache pool, CUDA graphs
```

Key files:
- `server/launch.py` — process spawning
- `scheduler/scheduler.py` — main loop (overlap_loop / normal_loop)
- `scheduler/io.py` — ZMQ I/O mixin (recv from tokenizer, send results)
- `engine/engine.py` — model init, forward_batch, KV cache pool
- `kvcache/mha_pool.py` — KV buffer: shape `(2, num_layers, num_pages, page_size, kv_heads, head_dim)`
- `message/` — msgpack-serialized dataclasses over ZMQ

---

## Target Architecture

```
Frontend (FastAPI)
  ↕ ZMQ
Tokenizer
  ↕ ZMQ
Router (new process)
  ├──→ ZMQ → P Worker (Prefill Scheduler)
  │              ↓ KV Transfer (NCCL send)
  └──→ ZMQ → D Worker (Decode Scheduler)
                 ↑ KV Transfer (NCCL recv)
  ↕ ZMQ
Detokenizer
  ↕ ZMQ
Frontend
```

### Roles

| Component | Responsibility |
|-----------|---------------|
| **Router** | Receives tokenized requests, dispatches to P worker; receives "prefill done" notification, tells D worker to expect KV; passes decode outputs to detokenizer |
| **P Worker** | Runs prefill only. After prefill, sends KV cache to D worker via NCCL, then notifies Router. Frees local KV. |
| **D Worker** | Receives KV from P worker, runs decode loop. Sends token outputs to detokenizer. |

---

## Detailed Design

### 1. New Message Types (`message/disagg.py`)

```python
@dataclass
class PrefillDoneMsg:
    """P worker → Router: prefill completed, KV ready to transfer"""
    uid: int
    seq_len: int            # total sequence length after prefill (= len(input_ids))
    first_token: int        # first sampled token from prefill
    num_pages: int          # number of KV pages to transfer
    sampling_params: SamplingParams
    input_ids: torch.Tensor  # full input_ids for D worker's Req

@dataclass
class StartDecodeMsg:
    """Router → D worker: start receiving KV for this request"""
    uid: int
    seq_len: int
    first_token: int
    num_pages: int
    sampling_params: SamplingParams
    input_ids: torch.Tensor

@dataclass
class TransferReadyMsg:
    """D worker → Router: KV received, decode can begin"""
    uid: int
```

### 2. KV Transfer Abstraction (`transfer/base.py`)

```python
class BaseKVTransfer(ABC):
    @abstractmethod
    def send_kv(self, kv_cache: BaseKVCachePool, page_indices: torch.Tensor,
                dst_rank: int, num_layers: int) -> None:
        """Send KV pages to destination. Called by P worker."""

    @abstractmethod
    def recv_kv(self, kv_cache: BaseKVCachePool, page_indices: torch.Tensor,
                src_rank: int, num_layers: int) -> None:
        """Receive KV pages into local cache. Called by D worker."""
```

**NCCL Implementation** (`transfer/nccl.py`):
- Uses a dedicated NCCL communicator (separate from TP group)
- P and D workers join a "transfer group" at startup
- `send_kv`: for each layer, gather KV from page_indices, nccl.send to D rank
- `recv_kv`: for each layer, nccl.recv into pre-allocated pages

For PoC: layer-by-layer send/recv (simple but not optimal). Optimization (pipelining, chunking) deferred.

### 3. P Worker (Prefill-Only Scheduler)

A stripped-down Scheduler that:
- Only has `PrefillManager` + `CacheManager` (no `DecodeManager`)
- After prefill forward:
  - Samples the first token (it already has the logits)
  - Extracts the KV page indices for the request
  - Calls `kv_transfer.send_kv(...)` to send KV to D worker
  - Sends `PrefillDoneMsg` to Router (includes first sampled token)
  - Frees local KV pages immediately

Key change from current Scheduler: `_process_last_data` does KV transfer + notify instead of adding req to DecodeManager.

### 4. D Worker (Decode-Only Scheduler)

A modified Scheduler that:
- Only has `DecodeManager` + `CacheManager` (no `PrefillManager`)
- Listens for `StartDecodeMsg` from Router
- Allocates local KV pages for the received sequence
- Calls `kv_transfer.recv_kv(...)` to receive KV into those pages
- Constructs a `Req` with `cached_len = seq_len` (all prefill KV already present), `device_len = seq_len + 1` (first token appended to input_ids)
- Adds req directly to DecodeManager
- Proceeds with normal decode loop (no redundant forward needed since P already sampled first token)

### 5. Router Process

Simple stateless dispatcher:
- Receives `UserMsg` from tokenizer, forwards to P worker
- Receives `PrefillDoneMsg` from P worker, sends `StartDecodeMsg` to D worker
- Receives `DetokenizeMsg` from D worker, forwards to detokenizer

For PoC: fixed 1P:1D mapping (no load balancing).

### 6. Launch Changes (`server/launch.py`)

New launch mode activated by `--pd-disagg` flag:

```python
# Instead of spawning N identical schedulers:
# 1. Spawn P worker(s)
# 2. Spawn D worker(s)
# 3. Spawn Router
# 4. Spawn tokenizer/detokenizer as before
```

New args in `ServerArgs`:
- `--pd-disagg`: enable PD disaggregation mode
- `--role`: `prefill` | `decode` | `router` (for per-process launch)
- `--transfer-group-addr`: NCCL init address for KV transfer group

---

## Implementation Steps

### Step 1: KV Transfer Layer
- Create `python/minisgl/transfer/__init__.py`, `base.py`, `nccl.py`
- Implement NCCL-based send/recv of KV pages between two ranks
- Unit test: two processes, one sends KV pages, other receives and verifies

### Step 2: New Message Types
- Add `message/disagg.py` with `PrefillDoneMsg`, `StartDecodeMsg`
- Register in serialization system

### Step 3: P Worker
- Create `python/minisgl/scheduler/prefill_worker.py`
- Fork from Scheduler, remove DecodeManager
- Override `_process_last_data` to do KV transfer instead of sampling

### Step 4: D Worker
- Create `python/minisgl/scheduler/decode_worker.py`
- Fork from Scheduler, remove PrefillManager
- Add handler for `StartDecodeMsg`: allocate pages → recv KV → create Req → add to DecodeManager

### Step 5: Router
- Create `python/minisgl/scheduler/router.py`
- Simple ZMQ relay: tokenizer → P worker, P worker → D worker, D worker → detokenizer

### Step 6: Launch Integration
- Add `--pd-disagg` to `server/args.py`
- Modify `server/launch.py` to spawn P/D/Router in disagg mode

### Step 7: End-to-End Test
- Script that launches 1P + 1D + Router, sends a request, verifies correct output

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `python/minisgl/transfer/__init__.py` | New — transfer abstraction |
| `python/minisgl/transfer/base.py` | New — `BaseKVTransfer` ABC |
| `python/minisgl/transfer/nccl.py` | New — NCCL implementation |
| `python/minisgl/message/disagg.py` | New — PD-specific messages |
| `python/minisgl/message/__init__.py` | Modify — export new messages |
| `python/minisgl/scheduler/prefill_worker.py` | New — P worker scheduler |
| `python/minisgl/scheduler/decode_worker.py` | New — D worker scheduler |
| `python/minisgl/scheduler/router.py` | New — Router process |
| `python/minisgl/server/args.py` | Modify — add `--pd-disagg` args |
| `python/minisgl/server/launch.py` | Modify — disagg launch path |
| `tests/core/test_pd_disagg.py` | New — E2E test |

---

## Verification

1. **Unit test KV transfer**: Two CUDA processes, P sends 1 layer of KV, D receives, assert equal
2. **E2E test**: Launch full disagg stack with `--dummy-weight`, send a greedy request, verify output matches non-disagg mode output for same input
3. **Manual smoke test**: Run with a small model (e.g., TinyLlama), confirm correct generation

---

## Open Questions / Deferred

- **Multiple P/D workers + load balancing** — deferred (PoC is 1:1)
- **Overlap KV transfer with next prefill** — deferred
- **Prefix cache sharing across P/D** — deferred
- **Fault tolerance** — deferred
- **TP within P or D worker** — the current TP design works unchanged within each worker; the transfer layer operates at the per-rank level (each TP rank transfers its own KV shard)
