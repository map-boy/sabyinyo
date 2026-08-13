import os
import torch
import torch.distributed as dist


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
