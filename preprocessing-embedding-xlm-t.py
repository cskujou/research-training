import pandas as pd
import torch
import numpy as np
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel

train_raw = pd.read_csv("train.csv")

split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
train, val= None, None
for train_idx, val_idx in split.split(train_raw, train_raw["language"]):
    train = train_raw.iloc[train_idx]
    val = train_raw.iloc[val_idx]

model_name = "cardiffnlp/twitter-xlm-roberta-base"
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
model = XLMRobertaModel.from_pretrained(model_name)


def text_to_embeding(t):
    ids = tokenizer.encode(t)
    out = model(torch.tensor(ids).unsqueeze(0))
    return out.pooler_output.detach().numpy()


X_train, y_train = [], []
for text, label in zip(train["text"], train["label"]):
    X_train.append(text_to_embeding(text))
    y_train.append(np.array([label]))
X_val, y_val = [], []
for text, label in zip(val["text"], val["label"]):
    X_val.append(text_to_embeding(text))
    y_val.append(np.array([label]))

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)

with open("pickle_xlmt", "wb") as f:
    pickle.dump((X_train, y_train, X_val, y_val), f)
