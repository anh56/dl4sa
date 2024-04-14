import torch.nn.functional as F
from modelGNN_updates import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        prob = [torch.softmax(logit) for logit in logits]

        if labels_list is not None:
            loss = 0
            for logit, labels in zip(logits, labels_list):
                loss_fct = nn.BCEWithLogitsLoss()
                num_labels = logit.shape[-1]
                loss += loss_fct(logit.view(-1, num_labels), labels.view(-1, num_labels))
            return loss, prob
        else:
            return prob


class GNNReGVD(nn.Module):
    def __init__(self, tune_params, encoder, config, tokenizer, args):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.tune_params = tune_params
        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer
        if args.gnn == "ReGGNN":
            self.gnn = ReGGNN(
                feature_dim_size=args.feature_dim_size,
                hidden_size=tune_params["hidden_size"],
                num_GNN_layers=tune_params["num_GNN_layers"],
                dropout=config.hidden_dropout_prob,
                residual=not args.remove_residual,
                att_op=tune_params["att_op"]
            )
        else:
            self.gnn = ReGCN(
                feature_dim_size=args.feature_dim_size,
                hidden_size=tune_params["hidden_size"],
                num_GNN_layers=tune_params["num_GNN_layers"],
                dropout=config.hidden_dropout_prob,
                residual=not args.remove_residual,
                att_op=tune_params["att_op"]
            )
        gnn_out_dim = self.gnn.out_dim
        self.classifier = MultitaskPredictionClassification(
            tune_params,
            config,
            args,
            input_size=gnn_out_dim
        )

    def forward(self, input_ids=None, labels_list=None):
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(
                input_ids.cpu().detach().numpy(),
                self.w_embeddings,
                window_size=self.tune_params["window_size"]
            )
        else:
            adj, x_feature = build_graph_text(
                input_ids.cpu().detach().numpy(),
                self.w_embeddings,
                window_size=self.tune_params["window_size"]
            )
        # initilizatioin
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        # run over GNNs
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())

        logits = self.classifier(outputs)  # logits is a list of tensors
        prob = [torch.softmax(logit) for logit in logits]

        if labels_list is not None:
            loss = 0
            for logit, labels in zip(logits, labels_list):  # Iterate over each tensor in logits and labels_list
                loss_fct = nn.BCEWithLogitsLoss()
                num_labels = logit.shape[-1]  # Get the number of labels for the current tensor
                loss += loss_fct(logit.view(-1, num_labels), labels.view(-1, num_labels))
            return loss, prob
        else:
            return prob


# modified from https://github.com/saikat107/Devign/blob/master/modules/model.py
class DevignModel(nn.Module):
    def __init__(self, tune_params, encoder, config, tokenizer, args):
        super(DevignModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer

        self.gnn = GGGNN(
            feature_dim_size=args.feature_dim_size,
            hidden_size=tune_params["hidden_size"],
            num_GNN_layers=tune_params["num_GNN_layers"],
            # num_classes=args.num_classes,
            dropout=config.hidden_dropout_prob
        )

        self.conv_l1 = torch.nn.Conv1d(tune_params["hidden_size"], tune_params["hidden_size"], 3).double()
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2).double()
        self.conv_l2 = torch.nn.Conv1d(tune_params["hidden_size"], tune_params["hidden_size"], 1).double()
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2).double()

        self.concat_dim = args.feature_dim_size + tune_params["hidden_size"]
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3).double()
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2).double()
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1).double()
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2).double()

        # self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=args.num_classes).double()
        # self.mlp_y = nn.Linear(in_features=args.hidden_size, out_features=args.num_classes).double()
        # [3, 3, 2, 3, 3, 3, 3]
        self.mlp_z = nn.ModuleList(
            [
                nn.Linear(in_features=self.concat_dim, out_features=num_labels).double()
                for num_labels in [3, 3, 2, 3, 3, 3, 3]
            ]
        )
        self.mlp_y = nn.ModuleList(
            [
                nn.Linear(in_features=tune_params["hidden_size"], out_features=num_labels).double()
                for num_labels in [3, 3, 2, 3, 3, 3, 3]
            ]
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, labels_list=None):
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings)
        # initilization
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature).to(device).double()
        # run over GGGN
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(),
                           adj_mask.to(device).double()).double()
        #
        c_i = torch.cat((outputs, adj_feature), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(nn.functional.relu(self.conv_l1(outputs.transpose(1, 2))))
        Y_2 = self.maxpool2(nn.functional.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(nn.functional.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(nn.functional.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)

        # iterate over the module list and apply each linear layer to the corresponding outputs
        logits = []
        for mlp_y_layer, mlp_z_layer in zip(self.mlp_y, self.mlp_z):
            before_avg_y = mlp_y_layer(Y_2)
            before_avg_z = mlp_z_layer(Z_2)
            before_avg = torch.mul(before_avg_y, before_avg_z)
            avg = before_avg.mean(dim=1)
            logits.append(avg)

        prob = [torch.softmax(logit) for logit in logits]

        if labels_list is not None:
            loss = 0
            for logit, labels in zip(logits, labels_list):  # Iterate over each tensor in logits and labels_list
                loss_fct = nn.BCEWithLogitsLoss()
                num_labels = logit.shape[-1]  # Get the number of labels for the current tensor
                loss += loss_fct(logit.view(-1, num_labels), labels.view(-1, num_labels))
            return loss, prob
        else:
            return prob
