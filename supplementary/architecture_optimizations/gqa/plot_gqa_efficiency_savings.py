# Companion to memory_estimator_gqa.py
# (1) Bar chart of KV-cache memory for MHA vs GQA across model configs.
# (2) Line plot of KV-cache memory saved (GB) vs group size for the same models.
#
# ---------------------------------------------------------------------------
# Modelling assumptions (kept consistent with memory_estimator_gqa.py and the
# GQA model code in gpt_with_kv_gqa.py):
#
#   * KV cache stores only the Key and Value tensors.  The factor of 2 in the
#     byte formula is (K, V).  Attention scores (Q @ K^T) and softmax weights
#     are transient -- recomputed every step -- so they are NOT cached and are
#     excluded from the estimate.
#   * head_dim = emb_dim // n_heads  (exact integer division).  The GQA model
#     asserts `d_out % num_heads == 0` (gpt_with_kv_gqa.py:63-68); we assert
#     the same and reject non-divisible configs instead of silently using
#     math.ceil like memory_estimator_gqa.py:25 does.
#   * MHA baseline: n_kv_heads = n_heads  (one K/V set per query head).
#   * GQA:          n_kv_heads = n_heads // group_size, where
#     group_size = number of query heads sharing one KV head = the MHA:GQA
#     compression ratio.  Assert n_heads % group_size == 0.
#   * Savings fraction = 1 - 1/group_size, which is model-INVARIANT (it depends
#     only on the sharing ratio, not on emb_dim / n_layers).  The earlier %
#     plot therefore overlapped into a single line.  Per-model gains are shown
#     as absolute GB saved, the only quantity that varies across models.
#   * 1 GB = 10^9 bytes (SI), matching memory_estimator_gqa.convert_bytes.  Note
#     the gpt_with_kv_*.py scripts report GiB (2^30) via torch.cuda; these
#     differ by ~7.4% and are not mixed here.
#
# Naming caveat: the swept `group_size` below equals the estimator's
# `--n_kv_groups` argument (the compression divisor), NOT the KV-head count.
# The GQA model code's `num_kv_groups` arg means the KV-head count instead
# (W_key width = num_kv_groups * head_dim) -- the two source scripts reuse the
# same name for opposite quantities.  We label the axis as group size to stay
# unambiguous.
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
from memory_estimator_gqa import calc_kv_bytes_total, DTYPE_BYTES


def to_gb(n_bytes):
    return n_bytes / (1000 ** 3)


def percent_savings(total_mha, total_gqa):
    return (1.0 - (total_gqa / total_mha)) * 100.0


# Representative model configs: (name, emb_dim, n_heads, n_layers).
# Every entry must satisfy emb_dim % n_heads == 0 (asserted at runtime).
MODELS = [
    ("GPT-2 (124M)",   768,  12, 12),
    ("LLaMA-2 (7B)",   4096, 32, 32),
    ("Gemma-2 (9B)",   3584, 16, 42),
    ("LLaMA-2 (70B)",  8192, 64, 80),
    ("Qwen2.5 (72B)",  8192, 64, 80),
]

# Group sizes (query heads per KV head = MHA:GQA compression ratio) to sweep.
# group_size=1 -> MHA (no sharing); group_size=n_heads -> MQA (one KV head).
GROUP_SIZES = [1, 2, 4, 8, 16, 32, 64]


def _check_model(name, emb_dim, n_heads):
    assert emb_dim % n_heads == 0, (
        f"{name}: emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads}) "
        f"so head_dim is exact"
    )


def plot_gqa_efficiency_all_models():
    context_length = 32768
    batch_size = 1
    dtype = "bf16"
    bytes_per_elem = DTYPE_BYTES[dtype]

    for name, emb_dim, n_heads, _ in MODELS:
        _check_model(name, emb_dim, n_heads)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"wspace": 0.45})
    ax_bar = axes[0]
    ax_line = axes[1]

    # ---- Left panel: absolute KV-cache memory, MHA vs GQA (group_size=4) ----
    group_size_fixed = 4
    names, mha_bytes, gqa_bytes = [], [], []
    for name, emb_dim, n_heads, n_layers in MODELS:
        assert n_heads % group_size_fixed == 0, (
            f"{name}: n_heads ({n_heads}) must be divisible by group_size "
            f"({group_size_fixed})"
        )
        n_kv_heads = n_heads // group_size_fixed
        total_mha = calc_kv_bytes_total(batch_size, context_length, emb_dim,
                                        n_heads, n_heads, n_layers, bytes_per_elem)
        total_gqa = calc_kv_bytes_total(batch_size, context_length, emb_dim,
                                        n_heads, n_kv_heads, n_layers, bytes_per_elem)
        names.append(name)
        mha_bytes.append(total_mha)
        gqa_bytes.append(total_gqa)

    mha_gb = [to_gb(b) for b in mha_bytes]
    gqa_gb = [to_gb(b) for b in gqa_bytes]

    x = range(len(names))
    width = 0.36
    bars_mha = ax_bar.bar([i - width / 2 for i in x], mha_gb, width, label="MHA (KV total)")
    bars_gqa = ax_bar.bar([i + width / 2 for i in x], gqa_gb, width, label="GQA (KV total)")
    for bar in bars_mha:
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for i, bar in enumerate(bars_gqa):
        pct = percent_savings(mha_bytes[i], gqa_bytes[i])
        ax_bar.text(bar.get_x() + bar.get_width(), bar.get_height() + 0.05,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=7.5,
                    color="#1f7a3d", rotation=90)

    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(names, fontsize=8)
    ax_bar.set_ylabel("Total KV cache (GB)")
    ax_bar.set_title("KV-cache memory: MHA vs GQA "
                     f"(group_size={group_size_fixed})")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_bar.legend(fontsize=8)
    ax_bar.margins(y=0.25)

    # ---- Right panel: memory saved (GB) vs group size (line plot) ----
    # Savings % = 1 - 1/group_size is model-invariant, so per-model curves are
    # plotted as absolute GB saved (the model-dependent quantity).
    for name, emb_dim, n_heads, n_layers in MODELS:
        sizes = [g for g in GROUP_SIZES if n_heads % g == 0]
        mem_saved_gb = []
        for g in sizes:
            n_kv_heads = n_heads // g
            total_mha = calc_kv_bytes_total(batch_size, context_length, emb_dim,
                                            n_heads, n_heads, n_layers, bytes_per_elem)
            total_gqa = calc_kv_bytes_total(batch_size, context_length, emb_dim,
                                            n_heads, n_kv_heads, n_layers, bytes_per_elem)
            mem_saved_gb.append(to_gb(total_mha - total_gqa))
        ax_line.plot(sizes, mem_saved_gb, marker="o", label=name)

    ax_line.set_xscale("log", base=2)
    ax_line.set_xticks(GROUP_SIZES)
    ax_line.set_xticklabels([str(g) for g in GROUP_SIZES])
    ax_line.set_xlabel("Group size (query heads per KV head)")
    ax_line.set_ylabel("KV-cache memory saved vs MHA (GB)")
    ax_line.set_title("KV-cache memory saved vs group size")
    ax_line.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_line.legend(fontsize=8, loc="lower right")
    ax_line.margins(y=0.15)

    fig.suptitle(
        f"GQA efficiency vs MHA — context_length={context_length}, "
        f"batch={batch_size}, dtype={dtype}",
        fontsize=11, y=0.98
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


if __name__ == "__main__":
    plot_gqa_efficiency_all_models()