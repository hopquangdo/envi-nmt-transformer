import yaml

def clean_tokens(tokens, bos_id, eos_id):
    tokens = tokens[1:]  # bỏ BOS
    if tokens and tokens[-1] == eos_id:
        tokens = tokens[:-1]
    return tokens

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)