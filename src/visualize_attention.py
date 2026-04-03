"""
Visualize cross-attention khi dịch câu mẫu PhoMT.
Chạy: python -m src.visualize_attention
"""
from src.utils import load_config
from src.utils.model_loader import load_model
from src.utils.visualize import translate_with_attention, plot_attention, plot_all_layers

# ══════════════════════════════════════════════
#  ĐẦU VÀO / ĐẦU RA — chỉnh ở đây
# ══════════════════════════════════════════════
SENTENCE   = "Two days of heavy rain, high winds, and numerous tornadoes caused major damage across multiple states."
OUTPUT     = "outputs/attention.png"        # ảnh layer cuối (avg-head)
ALL_LAYERS = True                            # True → xuất thêm ảnh tất cả layers
LAYER      = -1                              # -1 = layer cuối
HEAD       = None                            # None = trung bình tất cả heads
CONFIG     = "configs/config.yaml"
# ══════════════════════════════════════════════


def main():
    """
    Hàm chính để thực hiện dịch và vẽ biểu đồ Attention (Heatmap).

    Input Demo:
        SENTENCE (global): 'Two days of heavy rain...'
        CONFIG (global): 'configs/config.yaml'
    Output Demo:
        (Lưu các file ảnh .png vào thư mục outputs/)
    """
    cfg = load_config(CONFIG)
    model, tokenizer, device = load_model(
        checkpoint_path=cfg["checkpoint"]["best_model"],
        tokenizer_path=cfg["tokenizer"]["model_path"],
    )
    max_len = cfg["inference"].get("max_len", 128)

    print(f"\nInput : {SENTENCE}")

    src_tok, tgt_tok, attns, translation = translate_with_attention(
        model, tokenizer, SENTENCE, device, max_len=max_len
    )

    print(f"Output: {translation}")
    print(f"Layers: {len(attns)} | Heads: {attns[0].shape[0]}")
    print(f"Src tokens ({len(src_tok)}): {src_tok}")
    print(f"Tgt tokens ({len(tgt_tok)}): {tgt_tok}")

    plot_attention(src_tok, tgt_tok, attns,
                   layer=LAYER, head=HEAD, output_path=OUTPUT)

    if ALL_LAYERS:
        from pathlib import Path
        p = Path(OUTPUT)
        plot_all_layers(src_tok, tgt_tok, attns,
                        head=HEAD,
                        output_path=str(p.parent / f"{p.stem}_all_layers{p.suffix}"))


if __name__ == "__main__":
    main()
