import torch
import datasets
import pandas as pd
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import XLMRobertaPreTrainedModel
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer


class Attention(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.hidden_size = hidden_size

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        x = inputs
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)

        att_score = F.softmax(att, dim=1)
        outputs = x * att_score
        return outputs


class IntimacyNet(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config.problem_type = "regression"
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.attention = Attention(hidden_size=config.hidden_size)
        self.reg = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        states = self.attention(self.dropout(outputs[0]))
        logits = self.reg(self.dropout(states[:, 0, :]))

        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    train_raw = pd.read_csv("train_cleaned.csv")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in split.split(train_raw, train_raw["language"]):
        train = train_raw.iloc[train_idx]
        val = train_raw.iloc[val_idx]

    train_dict = {"label": train["label"], "text": train["text"]}
    val_dict = {"label": val["label"], "text": val["text"]}
    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)

    model_name = "cardiffnlp/twitter-xlm-roberta-base"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = IntimacyNet.from_pretrained(model_name, num_labels=1)
    metric = datasets.load_metric("pearsonr")
    training_args = TrainingArguments(
        output_dir="./checkpoint",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        learning_rate=8e-6,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch",
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
