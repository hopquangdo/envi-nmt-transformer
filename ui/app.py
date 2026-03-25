import os
import sys
import streamlit as st

# Setup đường dẫn tuyệt đối cho project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from src.utils import load_config
from src.utils.model_loader import load_model
from src.utils.translate import Translator

# @st.cache_resource(show_spinner="Đang tải model (chỉ lần đầu)...")
# def load_app_model():
#     config_path = os.path.join(root_dir, "configs/config.yaml")
#     config = load_config(config_path)
#
#     # Load model & tokenizer
#     model, tokenizer, device = load_model(
#         checkpoint_path=os.path.join(root_dir, config["checkpoint"]["best_model"]),
#         tokenizer_path=os.path.join(root_dir, config["tokenizer"]["model_path"])
#     )
#
#     return Translator(
#         model=model,
#         tokenizer=tokenizer,
#         device=device,
#         max_len=config["inference"]["max_len"]
#     )

st.title("Ứng dụng Dịch Máy Anh-Việt")

# translator = load_app_model()

# Danh sách câu mẫu
samples = [
    "Khác",
    "Machine translation is amazing.",
    "This is a sample sentence to test the model.",
    "I am learning how to build AI applications.",
    "Can you translate this for me?"
]
selected_sample = st.selectbox("Chọn câu mẫu:", samples)

default_text = "" if selected_sample == "Khác" else selected_sample

# 1 ô input
text_input = st.text_area("Nhập tiếng Anh:", value=default_text)

if st.button("Dịch") and text_input:
    # Gọi hàm dịch
    # translation = translator.translate(text_input)
    translation = "hi"
    # 1 ô output
    st.text_area("Kết quả:", value=translation, height=150)
