from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, List, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _shard_tensor_view(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard as a zero-copy view where possible.

    Like ``_shard_tensor`` but avoids ``.clone()``:
    - dim-0 chunk → contiguous view (zero-copy)
    - dim-1 chunk → must call ``.contiguous()`` (copies 1/n data)
    - vocab slice → contiguous view (zero-copy)
    - unshard'd → original tensor reference
    """
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim]
        return value.chunk(n, dim=0)[r]
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        # dim-1 chunk is non-contiguous; must materialize for H2D transfer
        return value.chunk(n, dim=1)[r].contiguous()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :]
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


class MergeAccumulator:
    """Device-agnostic accumulator for QKV merge, gate_up merge, and expert stacking.

    Collects partial tensors and emits completed (merged/stacked) tensors.
    Works on both CPU and GPU tensors — the merge ops (torch.cat/torch.stack)
    execute on whichever device the input tensors reside.
    """

    def __init__(self, num_experts: int | None, is_moe: bool):
        self._num_experts = num_experts
        self._is_moe = is_moe
        self._merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
        self._expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}

    def process(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Feed one (name, tensor) pair. Returns 0 or more completed (name, tensor) pairs."""
        # --- QKV / gate_up merge ---
        if (info := _get_merge_info(name)) is not None:
            merged_key, slot, all_slots = info
            self._merge_buf.setdefault(merged_key, {})[slot] = tensor
            if not all(s in self._merge_buf[merged_key] for s in all_slots):
                return []
            parts = [self._merge_buf[merged_key][s] for s in all_slots]
            del self._merge_buf[merged_key]
            out_name, out_tensor = merged_key, torch.cat(parts, dim=0)
        else:
            out_name, out_tensor = name, tensor

        # --- Expert stacking (MoE only) ---
        if self._is_moe and (expert_info := _get_expert_stack_info(out_name)) is not None:
            packed_key, expert_idx = expert_info
            slots = self._expert_buf.setdefault(packed_key, {})
            slots[expert_idx] = out_tensor
            if len(slots) != self._num_experts:
                return []
            experts = [slots[idx] for idx in range(self._num_experts)]
            del self._expert_buf[packed_key]
            return [(packed_key, torch.stack(experts, dim=0))]

        return [(out_name, out_tensor)]

    def flush(self) -> list[tuple[str, torch.Tensor]]:
        """Drain any remaining buffers. Asserts everything is complete."""
        assert (
            not self._merge_buf
        ), f"Incomplete merge groups in checkpoint: {list(self._merge_buf.keys())}"
        assert (
            not self._expert_buf
        ), f"Incomplete expert tensors in checkpoint: {list(self._expert_buf.keys())}"
        return []


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"


def load_sharded_by_file(
    model_path: str,
) -> Iterator[List[Tuple[str, torch.Tensor]]]:
    """Yield one batch of sharded CPU tensor *views* per safetensors file.

    Each batch is yielded while the ``safe_open`` context is still alive, so
    the caller **must** consume (e.g. ``batch_h2d``) the tensors before
    advancing the iterator — after that the mmap backing is released.

    Tensors are sharded with ``_shard_tensor_view`` (zero-copy where possible)
    but are **not** merged or expert-stacked — the caller is responsible for
    feeding them through a ``MergeAccumulator``.
    """
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    for file in tqdm(files, desc="Loading shards", disable=not tp_info.is_primary()):
        batch: List[Tuple[str, torch.Tensor]] = []
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                view = _shard_tensor_view(
                    name, raw, tp_info.rank, tp_info.size, config.num_kv_heads
                )
                assert view.is_contiguous(), f"Non-contiguous shard for {name}, shape={view.shape}"
                batch.append((name, view))
            # Yield inside the safe_open context so mmap stays alive
            yield batch
        # safe_open closes here → mmap released
