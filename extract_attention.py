import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Embedding, Bidirectional, LSTM, Dropout,
                          Flatten, RepeatVector, Permute, Multiply, Activation)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ====================== CONFIG ======================
CSV_FILE = "support_vectors_benign.csv"
MAX_LEN = 200
EMBEDDING_DIM = 100
TOP_K = 8
USE_DROPOUT = False
# ====================================================

# ====== Load Data ======
df = pd.read_csv(CSV_FILE)
texts = df['processed_func'].astype(str).tolist()

# ====== Tokenize ======
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
X = pad_sequences(sequences, maxlen=MAX_LEN)

# ====== Create dummy embedding matrix ======
embedding_matrix = np.random.uniform(-0.05, 0.05, (len(word_index) + 1, EMBEDDING_DIM))

# ====== Attention block ======
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    attention = Dense(1, activation='tanh')(inputs)  # (batch, time_steps, 1)
    attention = Flatten()(attention)                 # (batch, time_steps)
    attention = Activation('softmax', name='attention_vec')(attention)
    attention = RepeatVector(inputs.shape[-1])(attention)  # (batch, input_dim, time_steps)
    attention = Permute([2, 1])(attention)                 # (batch, time_steps, input_dim)
    output_attention_mul = Multiply()([inputs, attention]) # (batch, time_steps, input_dim)
    return output_attention_mul

# ====== BiLSTM model ======
def BiLSTM_network(MAX_LEN, EMBEDDING_DIM, word_index, embedding_matrix, use_dropout=False):
    inputs = Input(shape=(MAX_LEN,))
    embedding = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                          input_length=MAX_LEN, trainable=False)(inputs)
    bilstm_1 = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    bilstm_2 = Bidirectional(LSTM(64, return_sequences=True))(bilstm_1)
    atten_layer = attention_3d_block(bilstm_2)
    flatten = Flatten()(atten_layer)
    dense_1 = Dense(64, activation='relu')(Dropout(0.5)(flatten)) if use_dropout else Dense(64, activation='relu')(flatten)
    dense_2 = Dense(32)(dense_1)
    dense_3 = Dense(1, activation='sigmoid')(dense_2)
    model = Model(inputs=inputs, outputs=dense_3)
    return model

# ====== Build and compile model ======
model = BiLSTM_network(MAX_LEN, EMBEDDING_DIM, word_index, embedding_matrix, USE_DROPOUT)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ====== Model to extract attention ======
attention_model = Model(inputs=model.input, outputs=model.get_layer('attention_vec').output)

# ====== Predict attention vectors ======
attention_weights = attention_model.predict(X, batch_size=32)

# ====== Extract attention words ======
attention_words = []
for i in range(len(attention_weights)):
    attn = attention_weights[i]
    token_ids = X[i]
    # attn là vector (MAX_LEN,)
    top_idx = np.argsort(attn)[-TOP_K:]         # lấy TOP_K chỉ số có attention cao nhất
    words = [index_word.get(token_ids[j], "<UNK>") for j in top_idx]
    attention_words.append(words)

# ====== In kết quả mẫu ======
for i, words in enumerate(attention_words[:10]):
    print(f"[Sample {i+1}] Top-{TOP_K} attention words: {words}")

# ====== Lưu attention words ra file ======
output_df = pd.DataFrame({
    "sample_index": list(range(len(attention_words))),
    "attention_words": [", ".join(words) for words in attention_words]
})

output_df.to_csv("attention_words.csv", index=False)
print("✅ Attention words đã được lưu vào file: attention_words.csv")
