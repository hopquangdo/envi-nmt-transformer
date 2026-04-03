"""
src/utils/visualize.py
----------------------
Visualization cross-attention weights khi dịch một câu.

Cách dùng:
    from src.utils.visualize import translate_with_attention, plot_attention, plot_all_layers
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend, an toàn trên mọi môi trường
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Step 1: Dịch + thu thập cross-attention weights
# ---------------------------------------------------------------------------

@torch.inference_mode()
def translate_with_attention(
    model,
    tokenizer,
    text: str,
    device,
    max_len: int = 128,
) -> tuple[list[str], list[str], list[np.ndarray], str]:
    """
    Dịch một câu và trả về cross-attention weights của decoder.

    Input Demo:
        text: 'Hello world'
    Output Demo:
        src_tokens: ['Hello', 'world']
        tgt_tokens: ['Xin', 'chào', 'thế', 'giới']
        cross_attns: list[np.ndarray] -> mỗi phần tử shape (H, T_tgt, T_src)
        translation: 'Xin chào thế giới'
    """
    model.eval()

    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    pad_id = tokenizer.pad_id()

    # ---- Encode source ----
    ids = tokenizer.encode(text)
    src_ids = [bos_id] + ids[: max_len - 2] + [eos_id]
    src = torch.tensor([src_ids], device=device)
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)  # (1,1,1,T_src)

    enc_out = model.encoder(src, src_mask)

    # ---- Greedy decode ----
    tgt = torch.tensor([[bos_id]], device=device)
    for _ in range(max_len):
        T = tgt.size(1)
        tgt_mask = (
            torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            .unsqueeze(0).unsqueeze(0)
        )
        logits = model.decoder(tgt, enc_out, src_mask, tgt_mask)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        tgt = torch.cat([tgt, next_tok], dim=1)
        if next_tok.item() == eos_id:
            break

    # ---- Teacher-forced forward để lấy attention ----
    T_tgt = tgt.size(1)
    tgt_mask_full = (
        torch.tril(torch.ones(T_tgt, T_tgt, device=device, dtype=torch.bool))
        .unsqueeze(0).unsqueeze(0)
    )
    _, cross_attns_raw = model.decoder(
        tgt, enc_out, src_mask, tgt_mask_full, return_attn=True
    )
    # cross_attns_raw: list of (1, H, T_tgt, T_src) tensors

    # ---- Decode token strings ----
    # Source: bỏ BOS và EOS
    src_tokens = [tokenizer.id_to_piece(t) for t in src_ids[1:-1]]

    # Target: bỏ BOS ở đầu, dừng trước EOS
    tgt_ids_list = tgt.squeeze(0).tolist()  # [BOS, t1, ..., tn, EOS?]
    tgt_content = []
    for t in tgt_ids_list[1:]:  # bỏ BOS
        if t in (eos_id, pad_id):
            break
        tgt_content.append(t)
    tgt_tokens = [tokenizer.id_to_piece(t) for t in tgt_content]
    translation = tokenizer.decode(tgt_content)

    # Squeeze batch dim, convert to float32 numpy
    cross_attns = [a.squeeze(0).cpu().float().numpy() for a in cross_attns_raw]
    # cross_attns[i]: (H, T_tgt, T_src)

    return src_tokens, tgt_tokens, cross_attns, translation


# ---------------------------------------------------------------------------
# Step 2: Plot một layer
# ---------------------------------------------------------------------------

def plot_attention(
    src_tokens: list[str],
    tgt_tokens: list[str],
    cross_attns: list[np.ndarray],
    layer: int = -1,
    head: int | None = None,
    output_path: str = "outputs/attention.png",
    title: str | None = None,
) -> str:
    """
    Vẽ cross-attention heatmap cho một decoder layer và lưu ảnh.

    Input Demo:
        src_tokens: ['Hello', 'world']
        tgt_tokens: ['Xin', 'chào']
        cross_attns: list of arrays (H, T_tgt, T_src)
    Output Demo:
        return: '/absolute/path/to/outputs/attention.png'
    """
    n_layers = len(cross_attns)
    layer_idx = layer if layer >= 0 else n_layers + layer

    attn = cross_attns[layer]   # (H, T_tgt, T_src)
    n_heads = attn.shape[0]

    n_src = len(src_tokens)
    n_tgt = len(tgt_tokens)

    # Slice: bỏ vị trí 0 (BOS) ở cả tgt dim và src dim, bỏ EOS ở src dim
    # attn shape (H, T_tgt_full, T_src_full)
    # tgt_full = [BOS, t1..tn, EOS] → ta lấy positions 1..n_tgt
    # src_full = [BOS, w1..wm, EOS] → ta lấy positions 1..n_src
    attn_sliced = attn[
        :,
        1 : 1 + n_tgt,   # noqa: E203
        1 : 1 + n_src,   # noqa: E203
    ]  # (H, n_tgt, n_src)

    if head is None:
        attn_vis = attn_sliced.mean(axis=0)   # (n_tgt, n_src)
        head_label = f"avg {n_heads} heads"
    else:
        attn_vis = attn_sliced[head]           # (n_tgt, n_src)
        head_label = f"head {head}"

    # ---- Figure setup ----
    cell_w, cell_h = 0.65, 0.55
    fig_w = max(9, n_src * cell_w + 3.0)
    fig_h = max(6, n_tgt * cell_h + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f0e17")
    ax.set_facecolor("#1a1a2e")

    # Heatmap
    im = ax.imshow(
        attn_vis,
        cmap="YlOrRd",
        aspect="auto",
        vmin=0.0,
        vmax=float(attn_vis.max()) or 1.0,
        interpolation="nearest",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors="#e8e8e8", labelsize=8)
    cbar.set_label("Attention Weight", color="#e8e8e8", fontsize=9)
    cbar.outline.set_edgecolor("#444")

    # Axes
    ax.set_xticks(range(n_src))
    ax.set_yticks(range(n_tgt))
    ax.set_xticklabels(
        src_tokens, rotation=40, ha="right", fontsize=10,
        color="#7fdbca", fontfamily="monospace",
    )
    ax.set_yticklabels(
        tgt_tokens, fontsize=10,
        color="#f4c842", fontfamily="monospace",
    )
    ax.tick_params(colors="#aaa", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    ax.set_xlabel("Source (English)", color="#ccc", fontsize=12, labelpad=10)
    ax.set_ylabel("Target (Vietnamese)", color="#ccc", fontsize=12, labelpad=10)

    plot_title = title or (
        f"Cross-Attention  ·  Layer {layer_idx + 1}/{n_layers}  ·  {head_label}"
    )
    ax.set_title(plot_title, color="white", fontsize=14, fontweight="bold", pad=14)

    # Grid
    ax.set_xticks(np.arange(-0.5, n_src, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_tgt, 1), minor=True)
    ax.grid(which="minor", color="#333", linewidth=0.5)

    # Value annotations (chỉ khi bảng không quá lớn)
    if n_src * n_tgt <= 400:
        thresh = float(attn_vis.max()) * 0.55
        for i in range(n_tgt):
            for j in range(n_src):
                val = float(attn_vis[i, j])
                color = "#111" if val >= thresh else "#ddd"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold",
                )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅  Saved → {out.resolve()}")
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Step 3: Plot tất cả layers trong một figure
# ---------------------------------------------------------------------------

def plot_all_layers(
    src_tokens: list[str],
    tgt_tokens: list[str],
    cross_attns: list[np.ndarray],
    head: int | None = None,
    output_path: str = "outputs/attention_all_layers.png",
) -> str:
    """
    Vẽ cross-attention heatmap cho tất cả decoder layers trong một figure lưới (grid).

    Hắe hình được lưu thành file PNG với nền tối (dark mode).

    Args:
        src_tokens (list[str]): Danh sách token câu nguồn (tiếng Anh, đã bỏ BOS/EOS).
            Ví dụ: ['Hello', 'world'].
        tgt_tokens (list[str]): Danh sách token câu đích (tiếng Việt, đã bỏ BOS/EOS).
            Ví dụ: ['Xin', 'chào'].
        cross_attns (list[np.ndarray]): Danh sách N attention matrices, mỗi phần tử tương ứng
            với một decoder layer, shape (H, T_tgt_full, T_src_full).
        head (int | None): Chọn head cụ thể để hiển thị. None = trung bình tất cả heads.
        output_path (str): Đường dẫn file PNG đầu ra. Mặc định: 'outputs/attention_all_layers.png'.

    Returns:
        str: Đường dẫn tuyệt đối của file PNG đã lưu.
    """
    n_layers = len(cross_attns)
    n_src = len(src_tokens)
    n_tgt = len(tgt_tokens)

    ncols = min(3, n_layers)
    nrows = math.ceil(n_layers / ncols)

    cell_w, cell_h = 0.5, 0.45
    sub_w = max(5.0, n_src * cell_w + 1.5)
    sub_h = max(4.0, n_tgt * cell_h + 1.5)
    fig_w = ncols * sub_w
    fig_h = nrows * sub_h + 0.8

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("#0f0e17")

    for idx in range(n_layers):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor("#1a1a2e")

        attn = cross_attns[idx]             # (H, T_tgt, T_src)
        attn_sliced = attn[:, 1 : 1 + n_tgt, 1 : 1 + n_src]

        if head is None:
            attn_vis = attn_sliced.mean(axis=0)
            head_label = f"avg {attn.shape[0]}h"
        else:
            attn_vis = attn_sliced[head]
            head_label = f"h{head}"

        im = ax.imshow(
            attn_vis, cmap="YlOrRd", aspect="auto",
            vmin=0.0, vmax=float(attn_vis.max()) or 1.0,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(
            colors="#aaa", labelsize=6
        )

        ax.set_xticks(range(n_src))
        ax.set_yticks(range(n_tgt))
        ax.set_xticklabels(
            src_tokens, rotation=40, ha="right", fontsize=7,
            color="#7fdbca", fontfamily="monospace",
        )
        ax.set_yticklabels(
            tgt_tokens, fontsize=7,
            color="#f4c842", fontfamily="monospace",
        )
        ax.tick_params(colors="#aaa", length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        ax.set_xticks(np.arange(-0.5, n_src, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_tgt, 1), minor=True)
        ax.grid(which="minor", color="#333", linewidth=0.4)

        ax.set_title(
            f"Layer {idx + 1}  ·  {head_label}",
            color="white", fontsize=10, fontweight="bold", pad=6,
        )

    # Ẩn subplots thừa
    for idx in range(n_layers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Cross-Attention Weights — All Decoder Layers",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅  Saved → {out.resolve()}")
    return str(out.resolve())
