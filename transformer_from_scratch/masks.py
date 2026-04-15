import torch


def create_padding_mask(tokens: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    tokens: (batch_size, seq_len)

    returns:
        mask of shape (batch_size, 1, 1, seq_len)
        where 1 means keep and 0 means mask out
    """
    mask = (tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask.to(torch.int64)


def create_causal_mask(
    seq_len: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """
    returns:
        mask of shape (1, 1, seq_len, seq_len)
        where 1 means keep and 0 means mask out
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.int64))
    return mask.unsqueeze(0).unsqueeze(0)


def combine_masks(
    padding_mask: torch.Tensor, causal_mask: torch.Tensor
) -> torch.Tensor:
    """
    padding_mask: (batch_size, 1, 1, seq_len)
    causal_mask:  (1, 1, seq_len, seq_len)

    returns:
        combined mask: (batch_size, 1, seq_len, seq_len)
    """
    return padding_mask & causal_mask
