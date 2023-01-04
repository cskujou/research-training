import pickle
import pandas as pd
from laserembeddings import Laser
from sklearn.model_selection import StratifiedShuffleSplit

laser = Laser()
data = pd.read_csv("train_cleaned.csv")
data["language"].value_counts()
langs = {
    "English": "en",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh",
    "French": "fr",
    "Spanish": "es",
}

sentence = data["text"].tolist()
lang = data["language"].tolist()
lang = [langs[x] for x in lang]

X = laser.embed_sentences(
    sentences=sentence,
    lang=lang
)

y = data["label"].to_numpy()

split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
X_train, y_train, X_val, y_val = (None,) * 4
for train_idx, val_idx in split.split(X, lang):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

with open("pickle-laser", "wb") as f:
    pickle.dump((X_train, y_train, X_val, y_val), f)
