import random
from functools import lru_cache, partial

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
from triton.testing import do_bench

torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

data_type = torch.float16

# The kernels will utilize block sparisty to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def test_mask(
    score_mod=None,
    mask_mod=None,
    B=16,
    H=16,
    S=8192,
    D=64,
    skip_correctness=False,
    print_mask=True,
):
    assert (
        score_mod is not None or mask_mod is not None
    ), "Must provide a score_mod or mask_mod"
    query = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    value = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    gradOut = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    if mask_mod is not None:
        block_mask = create_block_mask(mask_mod, 1, 1, S, S, device=query.device)
    else:
        block_mask = None

    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=query.device)

    causal_fa2 = lambda: F.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    xformers_mask = lambda: F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )
    flex_attention_call = lambda: flex_attention(
        query, key, value, score_mod=score_mod, block_mask=block_mask
    )

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    # Forward pass
    causal_fa2_time = do_bench(causal_fa2)
    xformers_mask_time = do_bench(xformers_mask)
    flex_ms = do_bench(flex_attention_call)

    # Backward pass
    causal_fa2_out = causal_fa2()
    xformers_out = xformers_mask()
    flex_out = flex_attention_call()

    causal_fa2_bw_time = do_bench(
        lambda: causal_fa2_out.backward(gradOut, retain_graph=True)
    )
    xformers_mask_bw_time = do_bench(
        lambda: xformers_out.backward(gradOut, retain_graph=True)
    )
    flex_bw_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))

    # Inline correctness check
    if not skip_correctness:
        xformers_outs = []
        flex_outs = []

        query.grad = None
        key.grad = None
        value.grad = None

        out1 = xformers_mask()
        xformers_outs.append(out1)
        out1.backward(gradOut)
        xformers_outs += [query.grad, key.grad, value.grad]

        query.grad = None
        key.grad = None
        value.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [query.grad, key.grad, value.grad]
        for flex, xformer in zip(flex_outs, xformers_outs):
            torch.testing.assert_close(flex, xformer, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")
    # Usage in your results formatting:
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
        ],
        [
            "F.sdpa + mask",
            f"{xformers_mask_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_time, 4):.2f}",
            f"{xformers_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_bw_time, 10):.2f}",
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    print(
        f"\nResults for {score_mod.__name__ if score_mod is not None else mask_mod.__name__}:"
    )
    print(
        tabulate(
            results,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="grid",
        )
    )
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")

    # Clean up to save memory
    del query, key, value, gradOut, causal_fa2_out, xformers_out, flex_out
    torch.cuda.empty_cache()



num_block = 4  # number of temporal blocks
patch_block_size = 1024  # H*W
latent_block_size = 256  # number of latent tokens for each temporal block
# Total sequence length is num_block * patch_block_size + num_block * latent_block_size


def our_attn_mask(b, h, q_idx, kv_idx):
    """
    Attention mask for the following:
    Patch-patch attention: Block diagonal
    Patch-latent attention: Null
    Latent-patch attention: Block diagonal
    Latent-latent attention: Block causal
    where q, k, v are of format [patch_tokens] + [latent_tokens]
    with len(patch_tokens) = num_block * patch_block_size
    and len(latent_tokens) = num_block * latent_block_size
    """
    def is_from_patch_and_get_block_idx(idx):
        """
        First check if the index is from patch tokens.
        Return the patch_block_index if from patch tokens,
        otherwise return the latent_block_index.
        """
        is_from_patch = idx < num_block * patch_block_size
        block_idx = torch.where(
            is_from_patch,
            idx // patch_block_size,
            (idx - num_block * patch_block_size) // latent_block_size
        )
        return is_from_patch, block_idx
    
    q_is_from_patch, q_block_idx = is_from_patch_and_get_block_idx(q_idx)
    kv_is_from_patch, kv_block_idx = is_from_patch_and_get_block_idx(kv_idx)

    mask = torch.where(
        q_is_from_patch & kv_is_from_patch,
        q_block_idx == kv_block_idx,
        False,
    )
    mask = torch.where(
        q_is_from_patch & ~kv_is_from_patch,
        False,
        mask
    )
    mask = torch.where(
        ~q_is_from_patch & kv_is_from_patch,
        q_block_idx == kv_block_idx,
        mask
    )
    mask = torch.where(
        ~q_is_from_patch & ~kv_is_from_patch,
        q_block_idx >= kv_block_idx,
        mask
    )

    return mask


# PREFIX_LENGTH = 2048
# def prefix_lm_causal_mask(b, h, q_idx, kv_idx):
#     prefix_mask = kv_idx <= PREFIX_LENGTH
#     causal_mask = q_idx >= kv_idx
#     return prefix_mask | causal_mask

test_mask(
    mask_mod=our_attn_mask,
    B=2,
    H=16,
    S=num_block * (patch_block_size + latent_block_size),
    D=128,
    print_mask=True
)