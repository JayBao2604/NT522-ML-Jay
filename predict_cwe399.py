import torch
from torch.utils.data import SequentialSampler, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from linevul_main import TextDataset  # Dataset theo định nghĩa chuẩn của bạn
import os


def predict(args, model, tokenizer):
    test_dataset = TextDataset(tokenizer, args, file_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    logits = []
    y_trues = []

    print("Predicting:")
    for batch in tqdm(test_dataloader):
        if len(batch) == 3:
            input_ids, attention_mask, labels = [x.to(args.device) for x in batch]
        else:
            input_ids, labels = [x.to(args.device) for x in batch]
            attention_mask = input_ids.ne(1).long()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logit = outputs.logits

        logits.append(logit.cpu().numpy())
        y_trues.append(labels.cpu().numpy())

    # ====== Tính toán chỉ số đánh giá ======
    logits = np.concatenate(logits, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)

    if model.config.num_labels == 1:
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        y_preds = (probs > 0.3).astype(int).flatten()
    else:
        y_preds = np.argmax(logits, axis=1)

    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)

    print("\u2705 Đánh giá:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", type=str, default="original_test.csv")
    parser.add_argument("--model_path", type=str, default="linevul/model_02052025.bin")
    parser.add_argument("--tokenizer_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/codebert-base")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=512)  # ✅ thêm dòng này
    args = parser.parse_args()

    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 2  # hoặc 1 tùy vào mô hình đã train
    config.num_attention_heads = args.num_attention_heads

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device), strict=False)
    model.to(args.device)

    predict(args, model, tokenizer)


if __name__ == "__main__":
    main()
