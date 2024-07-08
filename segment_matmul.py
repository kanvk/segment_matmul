#!/usr/bin/env python

import torch
import sys

def seq_segment_matmul(inputs, ptr, other, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    sizes = (ptr.narrow(0, 1, ptr.numel() - 1) - ptr.narrow(0, 0, ptr.numel() - 1)).tolist()
    inputs_split = inputs.contiguous().split_with_sizes(sizes, 0)
    other_split = other.contiguous().split(1, 0)
    other_split = (other_split[0].squeeze(), other_split[1].squeeze())
    # print(inputs_split)
    # print(other_split)

    # assert torch.allclose(inputs[0:ptr[1]], inputs_split[0])
    # assert torch.allclose(inputs[ptr[1]: ptr[2]], inputs_split[1])

    M, K = inputs.shape
    P, K, N = other.shape
    c = torch.zeros((M, N), dtype=torch.float32).cuda()
    for mat in range(ptr.numel() - 1):
        a = inputs_split[mat]
        b = other_split[mat]
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        assert a.is_contiguous(), "matrix A must be contiguous"
        assert b.is_contiguous(), "matrix B must be contiguous"
        M, K = a.shape
        K, N = b.shape
        for m in range(0, M, BLOCK_SIZE_M):
            for n in range(0, N, BLOCK_SIZE_N):
                acc = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32).cuda()
                for k in range(0, K, BLOCK_SIZE_K):
                    A = a[m: m + BLOCK_SIZE_M, k: k + BLOCK_SIZE_K]
                    B = b[k: k + BLOCK_SIZE_K, n: n + BLOCK_SIZE_N]
                    acc += A @ B
                c[m + ptr[mat]: m + ptr[mat] + BLOCK_SIZE_M, n: n + BLOCK_SIZE_N] = acc
    return c


def main():
    torch.manual_seed(0)

    i = int(sys.argv[1]) * 128
    inputs = torch.randn(i, i).cuda()
    ptr = torch.tensor([0, i, i]).cuda()
    other = torch.randn(2, i, i).cuda()
    seq_out = seq_segment_matmul(inputs, ptr, other, i, i, i)
    print(seq_out)
    assert seq_out.size() == (i, i)


if __name__ == "__main__":
    main()
