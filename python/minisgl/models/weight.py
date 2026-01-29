from __future__ import annotations

import glob
import os
import time
from typing import Dict

import safetensors
import torch
from huggingface_hub import snapshot_download as hf_snapshot_download
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_up
from minisgl.utils.logger import init_logger
from modelscope import snapshot_download as ms_snapshot_download

logger = init_logger(__name__)
from tqdm.asyncio import tqdm


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue
        else:
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def load_weight(
    model_path: str,
    device: torch.device,
    source: str = "huggingface",
    use_mma: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Unified model weight loading function.
    :param source: "huggingface" or "modelscope"
    :param use_mma: Use MMA for accelerated CPU-GPU data transfer
    """
    if os.path.isdir(model_path):
        model_folder = model_path
    else:
        try:
            if source == "huggingface":
                model_folder = hf_snapshot_download(
                    model_path,
                    allow_patterns=["*.safetensors"],
                    tqdm_class=DisabledTqdm,
                )
            elif source == "modelscope":
                model_folder = ms_snapshot_download(
                    model_path,
                    allow_file_pattern=["*.safetensors"],
                )
            else:
                raise ValueError(
                    f"Unknown source: {source}, expected 'huggingface' or 'modelscope'"
                )
        except ValueError:
            raise
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid {source} model ID"
            )

    files = glob.glob(f"{model_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    # Log device transfer info
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    src_device = next(iter(state_dict.values())).device if state_dict else "N/A"
    logger.info(
        f"Transferring {len(state_dict)} tensors ({total_bytes / 1e9:.2f} GB) "
        f"from {src_device} to {device} (use_mma={use_mma})"
    )

    start_time = time.perf_counter()
    '''
    if use_mma:
        import mma

        if device.type == "cuda":
        
            torch.cuda.set_device(device)

    
        mma.init()
        new_state_dict = {}
        for k, v in state_dict.items():
            v = v.contiguous()
            gpu_tensor = torch.empty_like(v, device=device)
            if v.dtype == torch.bfloat16:
                cpu_data = v.view(torch.int16).numpy()
                mma.memcpy(gpu_tensor.view(torch.int16), cpu_data)
            else:
                cpu_data = v.numpy()
                mma.memcpy(gpu_tensor, cpu_data)
            torch.cuda.synchronize(device)
            new_state_dict[k] = gpu_tensor
        state_dict = new_state_dict

    else:
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
    '''

    if use_mma:
        import mma

        if device.type == "cuda":
            torch.cuda.set_device(device)

        # Pre-allocate GPU buffers with PyTorch first, then init MMA and memcpy.
        # This avoids "invalid resource handle" if MMA init changes CUDA/stream state.
        cpu_state_dict: Dict[str, torch.Tensor] = {}
        new_state_dict: Dict[str, torch.Tensor] = {}

        for k, v in state_dict.items():
            v = v.contiguous()
            cpu_state_dict[k] = v
            new_state_dict[k] = torch.empty_like(v, device=device)
        
        #start_time = time.perf_counter()
        mma.init()
        start_time = time.perf_counter()
        
        for k, v in cpu_state_dict.items():
            gpu_tensor = new_state_dict[k]
            if v.dtype == torch.bfloat16:
                cpu_data = v.view(torch.int16).numpy()
                mma.memcpy(gpu_tensor.view(torch.int16), cpu_data)
            else:
                cpu_data = v.numpy()
                mma.memcpy(gpu_tensor, cpu_data)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        state_dict = new_state_dict
    else:
        state_dict = {k: v.to(device) for k, v in state_dict.items()}

    elapsed = time.perf_counter() - start_time
    torch.cuda.set_stream(torch.cuda.default_stream(device))
    logger.info(f"Transfer completed in {elapsed:.2f}s ({total_bytes / elapsed / 1e9:.2f} GB/s)")
    return _merge_state_dict(state_dict)


# Backward compatibility
def load_hf_weight(
    model_path: str, device: torch.device, use_mma: bool = False
) -> Dict[str, torch.Tensor]:
    return load_weight(model_path, device, source="huggingface", use_mma=use_mma)


def load_ms_weight(
    model_path: str, device: torch.device, use_mma: bool = False
) -> Dict[str, torch.Tensor]:
    return load_weight(model_path, device, source="modelscope", use_mma=use_mma)
