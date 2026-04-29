"""
Analytical GPU memory estimator for transformer model training.

Estimates peak memory per GPU based on model architecture and parallelism
strategy. Used to predict OOM before running expensive simulations.

Memory components:
  1. Model parameters (fp16 weights)
  2. Gradients (fp16)
  3. Optimizer states (fp32 master weights + Adam momentum + variance)
  4. Activation memory (Megatron-LM formulation)

References
----------
[1] Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M.,
    Shoeybi, M., & Catanzaro, B. (2023).
    "Reducing Activation Recomputation in Large Transformer Models."
    MLSys 2023.  https://arxiv.org/abs/2205.05198
    — Activation memory formulas: Eq. 1 (no parallelism), Eq. 2 (tensor
      parallelism only), Eq. 3 (tensor + sequence parallelism).

[2] Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020).
    "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models."
    SC '20.  https://arxiv.org/abs/1910.02054
    — Mixed-precision weight/gradient/optimizer-state memory accounting
      (16 bytes per parameter) and ZeRO/FSDP sharding analysis (Table 1).

[3] Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J.,
    & Catanzaro, B. (2020).
    "Megatron-LM: Training Multi-Billion Parameter Language Models
    Using Model Parallelism."
    https://arxiv.org/abs/1909.08053
    — Tensor-parallel partitioning of attention and FFN parameters
      across TP ranks.

[4] Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M.,
    Korthikanti, V., ... & Catanzaro, B. (2021).
    "Efficient Large-Scale Language Model Training on GPU Clusters
    Using Megatron-LM."
    SC '21.  https://arxiv.org/abs/2104.04473
    — Pipeline-parallelism layer distribution (Section 3) and
      combined 3-D parallelism memory analysis.

[5] Touvron, H., et al. (2023).
    "LLaMA: Open and Efficient Foundation Language Models."
    https://arxiv.org/abs/2302.13971
    — Gated FFN (SwiGLU) architecture: 3 weight matrices per FFN
      layer instead of the standard 2.
"""

import math


def estimate_training_memory_per_gpu(
    vocab_size: int,
    dmodel: int,
    dff: int,
    num_heads: int,
    num_layers: int,
    batch_size_per_gpu: int,
    seq_len: int,
    dp: int,
    tp: int,
    pp: int,
    sp: int = 1,
    fsdp: bool = False,
    model_type: str = "gpt",
) -> float:
    """
    Estimate peak GPU memory (GB) for mixed-precision transformer training.

    Args:
        vocab_size: Vocabulary size.
        dmodel: Hidden dimension.
        dff: Feed-forward intermediate dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        batch_size_per_gpu: Per-GPU batch size.
        seq_len: Sequence length.
        dp: Data parallelism degree.
        tp: Tensor parallelism degree.
        pp: Pipeline parallelism degree.
        sp: Sequence parallelism degree.
        fsdp: Whether fully-sharded data parallelism is enabled.
        model_type: "gpt" for standard FFN, "llama"/"dense" for gated FFN.

    Returns:
        Estimated peak memory in GB.
    """
    # ---- 1. Model parameters per GPU [3][4] ----

    # Embedding table, divided across TP ranks [3]
    embedding_params = vocab_size * dmodel / tp

    # Per transformer layer
    # Attention (Q, K, V, output projections): 4 * dmodel^2, divided by TP [3]
    attention_params_per_layer = 4 * dmodel * dmodel / tp

    # FFN — gated architectures (LLaMA/dense) use 3 matrices [5], standard GPT uses 2 [3]
    if model_type in ("llama", "dense"):
        ffn_params_per_layer = 3 * dmodel * dff / tp
    else:
        ffn_params_per_layer = 2 * dmodel * dff / tp

    # Layer norms (small, not TP-divided)
    layernorm_params_per_layer = 4 * dmodel

    params_per_layer = (
        attention_params_per_layer + ffn_params_per_layer + layernorm_params_per_layer
    )

    # Pipeline parallelism distributes layers across stages [4]
    layers_per_gpu = math.ceil(num_layers / pp)
    total_layer_params = layers_per_gpu * params_per_layer

    # Embedding and output head live on the first and last PP stages
    # respectively [4, Section 3]. For a worst-case per-GPU estimate we
    # assume a single stage bears both (conservative).
    embedding_and_head_params = 2 * embedding_params

    params_per_gpu = total_layer_params + embedding_and_head_params

    # ---- 2. Weight / gradient / optimizer-state memory [2] ----
    # Mixed precision: fp16 weights (2B) + fp16 grads (2B)
    #   + fp32 master weights (4B) + momentum (4B) + variance (4B) = 16B  [2, Table 1]
    if fsdp:
        # FSDP/ZeRO shards gradients + optimizer states across DP ranks [2].
        # Peak = full fp16 weights (for compute) + sharded remainder.
        param_memory_bytes = params_per_gpu * (2 + 14.0 / dp)
    else:
        param_memory_bytes = params_per_gpu * 16

    # ---- 3. Activation memory per GPU [1] ----
    # Megatron-LM formulation (Korthikanti et al., 2023)
    s = seq_len
    b = batch_size_per_gpu
    h = dmodel
    a = num_heads

    if sp > 1 and tp > 1:
        s_eff = s / sp
        # Tensor + sequence parallelism: all activations divided by TP  [1, Eq. 3]
        act_per_layer = s_eff * b * h * (34 + 5 * a * s_eff / h) / tp
    elif tp > 1:
        # Tensor parallelism only  [1, Eq. 2]
        act_per_layer = s * b * h * (10 + 24.0 / tp + 5 * a * s / (h * tp))
    else:
        # No model parallelism  [1, Eq. 1]
        act_per_layer = s * b * h * (34 + 5 * a * s / h)

    activation_memory_bytes = layers_per_gpu * act_per_layer

    # ---- Total ----
    total_bytes = param_memory_bytes + activation_memory_bytes
    total_gb = total_bytes / (1024**3)

    return total_gb
