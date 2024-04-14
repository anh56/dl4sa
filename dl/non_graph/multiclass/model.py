import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        prob = torch.softmax(logits, -1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class CodeBERT(nn.Module):
    def __init__(self, tune_params, encoder, config, tokenizer, args):
        super(CodeBERT, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        prob = torch.softmax(logits, -1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class CNN(nn.Module):
    def __init__(
            self, tune_params, config, args, input_size=None,
    ):
        super().__init__()
        if input_size == None:
            input_size = config.hidden_size
        self.conv1d = nn.Conv1d(
            input_size, config.hidden_size,
            kernel_size=tune_params["kernel_size"], padding=tune_params["padding"]
        )
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, args.num_classes)

    def forward(self, x):
        x = x.unsqueeze(2)  # Add an extra dimension for the sequence length
        x = self.conv1d(x)
        x = torch.relu(x)
        # x = x.view(x.size(0), -1)  # Flatten the output for the dense layer
        x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model_CNN(nn.Module):
    def __init__(self, tune_params, encoder, config, tokenizer, args):
        super(Model_CNN, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = CNN(tune_params, config, args)

    def forward(self, input_ids=None, labels=None):
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput.logits
        logits = self.encoder(
            input_ids,
            attention_mask=input_ids.ne(1)
        ).logits
        logits = self.classifier(logits)
        prob = torch.softmax(logits, -1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class LSTM(nn.Module):
    def __init__(
            self, tune_params, config, args, input_size=None,
    ):
        super().__init__()
        if input_size == None:
            input_size = config.hidden_size
        self.lstm = nn.LSTM(input_size, config.hidden_size, batch_first=True, num_layers=tune_params["num_layers"])
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, args.num_classes)

    def forward(self, features):
        x = features
        x, _ = self.lstm(x.unsqueeze(0))
        x = x.squeeze(0)
        x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model_LSTM(nn.Module):
    def __init__(self, tune_params, encoder, config, tokenizer, args):
        # config.num_labels = config.hidden_size
        super(Model_LSTM, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = LSTM(tune_params, config, args)

    def forward(self, input_ids=None, labels=None):
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput.logits
        logits = self.encoder(
            input_ids,
            attention_mask=input_ids.ne(1)
        ).logits
        logits = self.classifier(logits)
        prob = torch.softmax(logits, -1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
