import yaml

def clean_tokens(tokens, bos_id, eos_id):
    """
    Loại bỏ các token đặc biệt <bos> và <eos> khỏi danh sách kết quả.

    Input Demo:
        tokens: [2, 15, 67, 3] (với bos_id=2, eos_id=3)
    Output Demo:
        return: [15, 67]
    """
    tokens = tokens[1:]  # bỏ BOS
    if tokens and tokens[-1] == eos_id:
        tokens = tokens[:-1]
    return tokens

def load_config(path):
    """
    Tải file cấu hình YAML (chứa các tham số huấn luyện).

    Input Demo:
        path: 'configs/config.yaml'
    Output Demo:
        return: dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)