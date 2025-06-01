import torch
import numpy as np
from sklearn.svm import SVC
from transformers import BertTokenizer, GPT2Tokenizer

# Khởi tạo tokenizer
def get_tokenizer(model_type="bert"):
    if model_type == "bert":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_type == "gpt2":
        return GPT2Tokenizer.from_pretrained("gpt2")
    return None

# Chuyển đổi văn bản thành vector
def text_to_vector(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return np.array(input_ids)

# Lấy chỉ số vector hỗ trợ từ SVM
def get_support_vector_idx(sample_vec, sample_label):
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(sample_vec, sample_label)
    return svclassifier.support_