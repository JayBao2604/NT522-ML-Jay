import pandas as pd
import time
import os
from google import genai  # Thay đổi từ openai sang genai
from google.genai import types  # Import thêm types
from string import Template
import random

# === Khởi tạo Gemini API client ===
GEMINI_API_KEY = ""  # Thay bằng API key của bạn
client = genai.Client(api_key=GEMINI_API_KEY)

# === Template prompt ===
PROMPT_TEMPLATE = Template("""
You are a professional C developer. Below is an original code snippet from a benign program:

$CONTEXT

Your task is:
1. Insert a short C code snippet (no more than 8 lines) that contains the following keywords: $attention_words.
2. The inserted code must be dead code (i.e., does not affect the behavior of the original code).
3. The code must be compilable, syntactically correct, and appear natural and legitimate.
4. Use meaningful names or mimic existing variable/function names in the context if possible.
5. Avoid I/O (e.g., printf), and preserve semantic stealthiness.

Please return the entire the inserted snippet.
""")

# === Common attention words theo category ===
common_words = {
    "data_type": ['int', 'float', 'double', 'char'],
    "control_statement": ['if', 'switch', 'for', 'while', 'do'],
    "storage_class": ['auto', 'extern', 'static', 'register'],
    "input_output": ['printf', 'scanf'],
    "misc": ['sizeof', 'return', 'break', 'continue']
}

# Gộp tất cả các common_words thành một list
all_common_words = []
for group in common_words.values():
    all_common_words.extend(group)

# === Đọc từ file top_attention_word_CWE399.csv ===
attention_word_df = pd.read_csv("top_attention_word_stats_js.csv")
top_words = attention_word_df['word'].dropna().unique().tolist()

# === Hợp nhất: top_words + common_words ===
all_attention_words = list(set(all_common_words + top_words))

# === Đọc file chứa các processed_func cần sinh mã đối kháng ===
support_vectors = pd.read_csv("output_js_new.csv")

# === Tạo thư mục output nếu chưa có ===
os.makedirs("adversarial_outputs_js", exist_ok=True)

# === Bắt đầu sinh mã đối kháng ===
for index, row in enumerate(support_vectors.iterrows()):
    if index >= 3000:
        break

    context = row[1]['processed_func']

    if not context or not isinstance(context, str) or context.strip() == "":
        print(f"[Skip] Index {index}: Empty context.")
        continue

    # Lấy ngẫu nhiên tối đa 3 attention words từ danh sách hợp nhất
    selected_words = random.sample(all_attention_words, k=min(3, len(all_attention_words)))
    selected_words_str = ', '.join(selected_words)

    # Tạo prompt hoàn chỉnh
    prompt = PROMPT_TEMPLATE.substitute(CONTEXT=context, attention_words=selected_words_str)

    try:
        # Gọi API của Gemini với cấu hình
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],  # Đặt nội dung trong danh sách
            config=types.GenerateContentConfig(
                max_output_tokens=10000,  # Số lượng token tối đa
                temperature=0.1  # Nhiệt độ (nếu cần)
            )
        )

        # Sử dụng response.text thay vì response.result
        generated_code = response.text.strip()  # Lấy kết quả từ response

        # Lưu ra file .c
        filename = f"adversarial_outputs_js/snippet_{index:04d}.js"
        with open(filename, "w") as f:
            f.write(generated_code)
        print(f"[Success] Index {index}: Snippet saved to {filename}")

    except Exception as e:
        print(f"[Error] Index {index}: {str(e)}")

    time.sleep(5)

print("✅ All adversarial snippets generated and saved.")
