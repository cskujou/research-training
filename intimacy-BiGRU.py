import logging
import os
import sys
import pickle
import time

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from scipy.stats import pearsonr

max_len = 512
num_epochs = 15
embed_size = 300
num_hiddens = 128
num_layers = 2
bidirectional = True
batch_size = 64
labels = 1
lr = 0.003
lstm_dropout = 0.1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
use_gpu = True


class SentimentNet(nn.Module):
    def __init__(
        self,
        embed_size,
        num_hiddens,
        num_layers,
        bidirectional,
        weight,
        labels,
        use_gpu,
        lstm_dropout,
        **kwargs,
    ):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.GRU(
            input_size=embed_size,
            hidden_size=self.num_hiddens,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=lstm_dropout,
            batch_first=True,
        )
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings)
        encoding = torch.cat([states[:, 0, :], states[:, -1, :]], dim=-1)
        outputs = self.decoder(encoding)
        return outputs


def pad_samples(features, maxlen=max_len, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while len(padded_feature) < maxlen:
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % "".join(sys.argv))

    logging.info("loading data...")
    with open("concept-net-embedding", "rb") as f:
        vectors, ids_to_tokens, tokens_to_ids = pickle.load(f)
    with open("pickle-concept-net", "rb") as f:
        X_train_ids, y_train, X_val_ids, y_val = pickle.load(f)
    logging.info("data loaded!")

    X_train_ids = torch.tensor(pad_samples(X_train_ids))
    X_val_ids = torch.tensor(pad_samples(X_val_ids))

    y_train = torch.tensor(y_train).unsqueeze(-1)
    y_val = torch.tensor(y_val).unsqueeze(-1)

    net = SentimentNet(
        embed_size=embed_size,
        num_hiddens=num_hiddens,
        num_layers=num_layers,
        bidirectional=bidirectional,
        weight=vectors,
        labels=labels,
        use_gpu=use_gpu,
        lstm_dropout=lstm_dropout,
    )
    net.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.NAdam(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(X_train_ids, y_train)
    val_set = torch.utils.data.TensorDataset(X_val_ids, y_val)

    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_iter = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc="Epoch %d" % epoch) as pbar:
            for feature, label in train_iter:
                n += 1
                net.zero_grad()
                feature = Variable(feature.to(device))
                label = Variable(label.to(device))
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_loss += loss

                pbar.set_postfix(
                    {
                        "epoch": f"{epoch}",
                        "train loss": f"{train_loss.data / n:.4f}",
                    }
                )
                pbar.update(1)

        with tqdm(total=len(val_iter), desc="Epoch %d" % epoch) as pbar:
            y_pred, y_real = [], []
            with torch.no_grad():
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.to(device)
                    val_label = val_label.to(device)
                    val_score = net(val_feature)
                    val_loss = loss_function(val_score, val_label)
                    y_pred.append(val_score)
                    y_real.append(val_label)
                    val_losses += val_loss

                    pbar.set_postfix(
                        {
                            "epoch": f"{epoch}",
                            "train loss": f"{train_loss.data / n:.4f}",
                            "val loss": f"{val_losses.data / m:.4f}",
                        }
                    )
                    pbar.update(1)
            y_pred = torch.cat(y_pred).cpu().view(-1).numpy()
            y_real = torch.cat(y_real).cpu().view(-1).numpy()
            r = pearsonr(y_pred, y_real)
            end = time.time()
            runtime = end - start
            pbar.set_postfix(
                {
                    "epoch": f"{epoch}",
                    "train loss": f"{train_loss.data / n:.4f}",
                    "val loss": f"{val_losses.data / m:.4f}",
                    "pearsonr": f"{r[0]:.4f}",
                    "time": f"{runtime:.2f}",
                }
            )
