import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from text_to_uri import standardized_uri
import emoji
import jieba

vocab = pd.read_hdf("mini.h5")

ids_to_tokens = vocab.index.tolist()
ids_to_tokens = ["<pad>", "<unk>"] + ids_to_tokens
vectors = vocab.to_numpy()
pad_vec = np.zeros(300)
unk_vec = vectors.mean(axis=0)
vectors = np.concatenate([pad_vec.reshape(1, 300), unk_vec.reshape(1, 300), vectors])

vectors = torch.tensor(vectors, dtype=torch.float)

tokens_to_ids = {}
with tqdm(total=len(ids_to_tokens)) as pbar:
    for index, token in enumerate(ids_to_tokens):
        tokens_to_ids[token] = index
        _ = pbar.update(1)

with open("concept-net-embedding", "wb") as f:
    pickle.dump((vectors, ids_to_tokens, tokens_to_ids), f)

vectors = vectors.numpy()

data = pd.read_csv("train_cleaned.csv")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train, val = None, None
for train_idx, val_idx in split.split(data, data["language"]):
    train = data.iloc[train_idx]
    val = data.iloc[val_idx]

langs = {
    "English": "en",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh",
    "French": "fr",
    "Spanish": "es",
}


def add_token(row, X_ids):
    text = emoji.demojize(row.text)
    ids = []
    lang = langs[row.language]
    if lang == "zh":
        zh_stop_words = set()
        with open("zh_stop_words", "r", encoding="UTF-8") as f:
            for w in f.readline():
                zh_stop_words.add(w)
        text = [w for w in jieba.lcut(text) if w not in zh_stop_words]

    for word in text:
        std_word = standardized_uri(lang, word)
        if std_word in tokens_to_ids:
            ids.append(tokens_to_ids[std_word])
        else:
            ids.append(tokens_to_ids["<unk>"])
    X_ids.append(ids)


X_train_ids, X_val_ids = [], []
with tqdm(total=len(train), desc="train") as pbar:
    for i, row in train.iterrows():
        add_token(row, X_train_ids)
        _ = pbar.update(1)
with tqdm(total=len(val), desc="val") as pbar:
    for i, row in val.iterrows():
        add_token(row, X_val_ids)
        _ = pbar.update(1)

y_train = train["label"].to_list()
y_val = val["label"].to_list()

with open("pickle-concept-net", "wb") as f:
    pickle.dump((X_train_ids, y_train, X_val_ids, y_val), f)

X_train, X_val = [], []
with tqdm(total=len(X_train_ids)) as pbar:
    for sentence in X_train_ids:
        avg = np.zeros(300)
        len_ = len(sentence)
        for ids in sentence:
            avg = vectors[ids] + avg
        X_train.append(avg / len_)
        _ = pbar.update(1)

with tqdm(total=len(X_val_ids)) as pbar:
    for sentence in X_val_ids:
        avg = np.zeros(300)
        len_ = len(sentence)
        for ids in sentence:
            avg = vectors[ids] + avg
        X_val.append(avg / len_)
        _ = pbar.update(1)

X_train = np.array(X_train)
X_val = np.array(X_val)

y_train = np.array(y_train)
y_val = np.array(y_val)

with open("pickle-concept-net-avg", "wb") as f:
    pickle.dump((X_train, y_train, X_val, y_val), f)
