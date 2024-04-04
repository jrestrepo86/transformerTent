import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from treet.attention import FixedPastCausalAttention as fpca
from treet.attention import get_mask


def test():
    N = 2
    model_dim = 6
    query_len = 3
    values_a = torch.arange(model_dim, dtype=torch.float).expand(
        [N, query_len, model_dim]
    )
    keys_a = torch.arange(N * model_dim * query_len, dtype=torch.float).reshape(
        N, query_len, model_dim
    )
    query_a = torch.ones_like(keys_a)

    values_b = torch.arange(start=10, end=16, dtype=torch.float).expand(
        [N, query_len, model_dim]
    )
    keys_b = 2.0 * torch.ones_like(keys_a)
    query_b = torch.arange(start=10, end=10 + model_dim, dtype=torch.float).expand(
        [N, query_len, model_dim]
    )

    mask = get_mask(N, keys_a.shape[1], query_a.shape[1], history_len=1)

    parameters = {"model_dim": 6, "heads": 2, "history_len": 1, "mask": mask}
    attention = fpca(**parameters)

    a = attention(values_a, keys_a, query_a)
    b = attention(values_b, keys_b, query_b, ref_sample=True)
    print(a)
    print(b)


if __name__ == "__main__":
    test()
