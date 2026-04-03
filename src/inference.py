import argparse

from src.utils import load_config
from src.utils.translate import Translator
from src.utils.model_loader import load_model


def main(config_path: str):
    """
    Hàm khởi chạy giao dịch dịch văn bản tương tác qua terminal.

    Input Demo:
        config_path: 'configs/config.yaml'
    Output Demo:
        (Hiển thị kết quả dịch trực tiếp trên terminal)
    """
    config   = load_config(config_path)
    inf_cfg  = config["inference"]
    ckpt_cfg = config["checkpoint"]
    tok_cfg  = config["tokenizer"]

    model, tokenizer, device = load_model(
        checkpoint_path=ckpt_cfg["best_model"],
        tokenizer_path=tok_cfg["model_path"],
    )

    translator = Translator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=inf_cfg["max_len"],
    )

    method    = inf_cfg.get("method", "beam")
    beam_size = inf_cfg.get("beam_size", 5)

    print("=" * 50)
    print(f"Method: {method} | Beam size: {beam_size}")
    print("Nhập câu để dịch (enter để thoát)")
    print("=" * 50)

    while True:
        text = input("Input: ")
        if text.strip() == "":
            break

        output = translator.translate(
            text,
            method=method,
            beam_size=beam_size,
        )

        print("Output:", output)
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMT inference")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Đường dẫn file config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)