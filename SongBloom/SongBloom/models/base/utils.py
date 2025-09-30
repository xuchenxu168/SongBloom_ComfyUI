import torch
import torch.nn as nn
import typing as tp

def length_to_mask(lengths: torch.Tensor, max_len: tp.Optional[int] = None) -> torch.Tensor:
    """Utility function to convert a tensor of sequence lengths to a mask (useful when working on padded sequences).
    For example: [3, 5] => [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    Args:
        lengths (torch.Tensor): tensor with lengths
        max_len (int): can set the max length manually. Defaults to None.
    Returns:
        torch.Tensor: mask with 0s where there is pad tokens else 1s
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length)[None, :].to(lengths.device) < lengths[:, None]


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    # phase = phase.to(torch.bfloat16)
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module for transformer encoder layer.

    Args:
        norm_type (str): Normalization method.
        dim (int): Dimension of the normalized layer.
        **kwargs (dict): Additional parameters for normalization layer.
    Returns:
        nn.Module: Normalization module.
    """
    if norm_type == 'layer_norm':
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")