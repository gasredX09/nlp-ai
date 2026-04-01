import torch


def print_tensor_info(name: str, x: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
