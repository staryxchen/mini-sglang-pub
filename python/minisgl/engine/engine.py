from __future__ import annotations

import concurrent.futures
import time
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.models.weight import MergeAccumulator, load_sharded_by_file
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

try:
    import mma as _mma_lib

    _MMA_AVAILABLE = True
except ImportError:
    _mma_lib = None
    _MMA_AVAILABLE = False

_mma_initialized = False


def _ensure_mma_init():
    global _mma_initialized
    if not _mma_initialized:
        t0 = time.perf_counter()
        _mma_lib.init()
        _mma_initialized = True
        logger.info(f"MMA initialized in {time.perf_counter() - t0:.3f}s")


logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        t_load_start = time.perf_counter()
        state_dict = self._load_weight_state_dict(config)
        t_state_dict = time.perf_counter()
        logger.info_rank0(f"_load_weight_state_dict took {t_state_dict - t_load_start:.2f}s")
        self.model.load_state_dict(state_dict)
        del state_dict
        t_load_end = time.perf_counter()
        logger.info_rank0(f"model.load_state_dict took {t_load_end - t_state_dict:.2f}s")
        logger.info_rank0(f"Total weight loading took {t_load_end - t_load_start:.2f}s")

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            if not config.use_mma:
                return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}
            else:
                # MMA-accelerated weight loading
                if not _MMA_AVAILABLE:
                    logger.warning(
                        "MMA requested but not available, falling back to default loading"
                    )
                    return {
                        k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)
                    }

                # Phase 0: Start MMA init in background (overlaps with first file mmap read)
                mma_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                mma_future = mma_executor.submit(_ensure_mma_init)

                t0 = time.perf_counter()
                accumulator = MergeAccumulator(
                    num_experts=config.model_config.num_experts,
                    is_moe=config.model_config.is_moe,
                )
                state_dict: Dict[str, torch.Tensor] = {}
                total_bytes = 0
                total_tensors = 0
                t_mma_wait = 0.0
                t_alloc = 0.0
                t_h2d = 0.0
                t_merge = 0.0
                file_idx = 0

                shard_iter = load_sharded_by_file(config.model_path)
                t_iter_init = time.perf_counter()
                logger.info_rank0(f"MMA: iterator init took {t_iter_init - t0:.2f}s")

                for sharded_batch in shard_iter:
                    t_file_start = time.perf_counter()

                    # Ensure MMA is ready before the first batch_h2d
                    if mma_future is not None:
                        mma_future.result()
                        mma_executor.shutdown(wait=False)
                        mma_future = None
                        t_mma_waited = time.perf_counter()
                        t_mma_wait = t_mma_waited - t_file_start
                        logger.info_rank0(f"MMA: init wait took {t_mma_wait:.2f}s")
                        t_file_start = time.perf_counter()  # reset for first file

                    names = [name for name, _ in sharded_batch]
                    cpu_tensors = [t for _, t in sharded_batch]

                    batch_bytes = 0
                    for t in cpu_tensors:
                        batch_bytes += t.numel() * t.element_size()
                    total_bytes += batch_bytes
                    total_tensors += len(cpu_tensors)

                    # Batch GPU alloc: one contiguous buffer, then split into views
                    t_a = time.perf_counter()
                    flat_buf = torch.empty(batch_bytes, dtype=torch.uint8, device=self.device)
                    gpu_tensors = []
                    offset = 0
                    for ct in cpu_tensors:
                        nbytes = ct.numel() * ct.element_size()
                        gpu_t = flat_buf[offset : offset + nbytes].view(ct.dtype).reshape(ct.shape)
                        gpu_tensors.append(gpu_t)
                        offset += nbytes
                    t_b = time.perf_counter()
                    t_alloc += t_b - t_a

                    # Batch H2D: mmap views → GPU (only 1/TP data)
                    if cpu_tensors:
                        _mma_lib.batch_h2d(gpu_tensors, cpu_tensors)
                    t_c = time.perf_counter()
                    t_h2d += t_c - t_b

                    # Merge/stack on GPU, then dtype conversion
                    for name, gpu_t in zip(names, gpu_tensors):
                        for final_name, final_t in accumulator.process(name, gpu_t):
                            state_dict[final_name] = final_t.to(self.dtype)
                    t_d = time.perf_counter()
                    t_merge += t_d - t_c

                    if file_idx < 3:
                        logger.info_rank0(
                            f"MMA: file[{file_idx}] alloc={t_b - t_a:.3f}s"
                            f" h2d={t_c - t_b:.3f}s merge={t_d - t_c:.3f}s"
                            f" total={t_d - t_file_start:.3f}s"
                            f" ({len(cpu_tensors)} tensors)"
                        )
                    file_idx += 1

                    del gpu_tensors, cpu_tensors, flat_buf

                # Flush any remaining merge/expert buffers
                for final_name, final_t in accumulator.flush():
                    state_dict[final_name] = final_t.to(self.dtype)

                t1 = time.perf_counter()
                logger.info_rank0(
                    f"MMA: GPU-side shard/merge pipeline took {t1 - t0:.2f}s"
                    f" ({total_tensors} tensors, {total_bytes / 1e9:.2f} GB transferred)"
                )
                logger.info_rank0(
                    f"MMA: breakdown: mma_wait={t_mma_wait:.2f}s"
                    f" alloc={t_alloc:.2f}s h2d={t_h2d:.2f}s"
                    f" merge={t_merge:.2f}s"
                    f" overhead={t1 - t0 - t_mma_wait - t_alloc - t_h2d - t_merge:.2f}s"
                )
                return state_dict

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
