import torch.nn as nn
import torch


# change the head classifier to a custom one to support multitask
class MultitaskPredictionClassification(nn.Module):
    def __init__(
            self, tune_params, config, args, input_size=None,
            num_labels_per_category: list[int] = [3, 3, 2, 3, 3, 3, 3]
    ):
        super().__init__()
        if input_size is None:
            input_size = tune_params["hidden_size"]
        self.dense = nn.Linear(input_size, tune_params["hidden_size"])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.ModuleList(
            [
                nn.Linear(tune_params["hidden_size"], num_labels)
                for num_labels in num_labels_per_category
            ]
        )

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = [layer(x) for layer in self.out_proj]
        return logits


class Model(nn.Module):
    def __init__(self, tune_params, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = MultitaskPredictionClassification(tune_params, config, args)

    def forward(self, input_ids=None, labels_list=None):
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput.logits
        logits = self.encoder(
            input_ids,
            attention_mask=input_ids.ne(1)
        ).logits
        logits = self.classifier(logits)
        prob = [torch.softmax(logit, -1) for logit in logits]

        if labels_list is not None:
            loss = 0
            for logit, labels in zip(logits, labels_list):
                loss_fct = nn.BCEWithLogitsLoss()
                num_labels = logit.shape[-1]
                loss += loss_fct(logit.view(-1, num_labels), labels.view(-1, num_labels))
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
        self.classifier = MultitaskPredictionClassification(tune_params, config, args)

    def forward(self, input_ids=None, labels_list=None):
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput.logits
        logits = self.encoder(
            input_ids,
            attention_mask=input_ids.ne(1)
        ).logits
        logits = self.classifier(logits)
        prob = [torch.softmax(logit, -1) for logit in logits]

        if labels_list is not None:
            loss = 0
            for logit, labels in zip(logits, labels_list):
                loss_fct = nn.BCEWithLogitsLoss()
                num_labels = logit.shape[-1]
                loss += loss_fct(logit.view(-1, num_labels), labels.view(-1, num_labels))
            return loss, prob
        else:
            return prob


class CNN(nn.Module):
    def __init__(
            self, tune_params, config, args, input_size=None,
            num_labels_per_category: list[int] = [3, 3, 2, 3, 3, 3, 3]
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
        self.out_proj = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, num_labels)
                for num_labels in num_labels_per_category
            ]
        )

    def forward(self, x):
        x = x.unsqueeze(2)  # Add an extra dimension for the sequence length
        x = self.conv1d(x)
        x = torch.relu(x)
        # x = x.view(x.size(0), -1)  # Flatten the output for the dense layer
        x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.dropout(x)
        logits = [layer(x) for layer in self.out_proj]
        return logits


class Model_CNN(nn.Module):
    def __init__(self, tune_params, encoder, config, tokenizer, args):
        # config.num_labels = config.hidden_size
        super(Model_CNN, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = CNN(tune_params, config, args)

    def forward(self, input_ids=None, labels_list=None):
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput.logits
        logits = self.encoder(
            input_ids,
            attention_mask=input_ids.ne(1)
        ).logits
        # logits = logits.transpose(0, 1)
        logits = self.classifier(logits)
        prob = [torch.softmax(logit, -1) for logit in logits]

        if labels_list is not None:
            loss = 0
            for logit, labels in zip(logits, labels_list):
                loss_fct = nn.BCEWithLogitsLoss()
                num_labels = logit.shape[-1]
                loss += loss_fct(logit.view(-1, num_labels), labels.view(-1, num_labels))
            return loss, prob
        else:
            return prob


class LSTM(nn.Module):
    def __init__(
            self, tune_params, config, args, input_size=None,
            num_labels_per_category: list[int] = [3, 3, 2, 3, 3, 3, 3]
    ):
        super().__init__()
        if input_size == None:
            input_size = config.hidden_size
        self.lstm = nn.LSTM(input_size, config.hidden_size, batch_first=True)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, num_labels)
                for num_labels in num_labels_per_category
            ]
        )

    def forward(self, features):
        x = features
        x, _ = self.lstm(x.unsqueeze(0))
        x = x.squeeze(0)
        x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.dropout(x)
        logits = [layer(x) for layer in self.out_proj]
        return logits


class Model_LSTM(nn.Module):
    def __init__(self, tune_params, encoder, config, tokenizer, args):
        # config.num_labels = config.hidden_size
        super(Model_LSTM, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = LSTM(tune_params, config, args)

    def forward(self, input_ids=None, labels_list=None):
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput.logits
        logits = self.encoder(
            input_ids,
            attention_mask=input_ids.ne(1)
        ).logits
        logits = self.classifier(logits)
        prob = [torch.softmax(logit, -1) for logit in logits]

        if labels_list is not None:
            loss = 0
            for logit, labels in zip(logits, labels_list):
                loss_fct = nn.BCEWithLogitsLoss()
                num_labels = logit.shape[-1]
                loss += loss_fct(logit.view(-1, num_labels), labels.view(-1, num_labels))
            return loss, prob
        else:
            return prob
