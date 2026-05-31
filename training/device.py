import torch


def setup_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[设备] {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"
        print("[设备] CPU")
    return device


def print_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU 显存: {allocated:.2f} GB")
