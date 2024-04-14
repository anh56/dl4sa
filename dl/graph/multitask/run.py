from __future__ import absolute_import, division, print_function

from datetime import datetime
import time
import argparse
import logging
import os
import random
import re
import shutil
import numpy as np
import scipy
import torch
from neptune.utils import stringify_unsupported
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from tqdm import tqdm, trange
import multiprocessing
from model import *
import ray
from ray import tune
from ray import train as ray_train
from ray.tune.schedulers import ASHAScheduler
import neptune
from neptune_pytorch import NeptuneLogger

cpu_cont = multiprocessing.cpu_count()
from transformers import (
    WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
    BertConfig, BertForMaskedLM, BertTokenizer,
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

# enable ray
ray.init(
    num_gpus=1,
    # dashboard_port=8265,
    # dashboard_host="0.0.0.0",
    # include_dashboard=True
)


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(
            self,
            input_tokens,
            input_ids,
            idx,
            label,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


class MultitaskInputFeatures(object):
    def __init__(
            self,
            input_tokens,
            input_ids,
            idx,
            labels,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.labels = labels


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    # specify the number of labels for each class
    # authentication only have 2 labels
    num_labels_per_class = [3, 3, 2, 3, 3, 3, 3]
    labels = [torch.zeros(num_labels) for num_labels in num_labels_per_class]
    target_cols = ["access_vector", "access_complexity", "authentication",
                   "confidentiality", "integrity", "availability", "severity"]

    for idx, col in enumerate(target_cols):
        label: int = int(js[col])
        labels[idx][label] = 1

    return MultitaskInputFeatures(
        source_tokens,
        source_ids,
        js['idx'],
        labels
    )


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

        if 'train' in file_path:
            logger.info("*** Total Sample ***")
            logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** First 3 Sample from Training dataset ***")
                logger.info("Total sample {}".format(idx))
                logger.info("idx: {}".format(idx))
                logger.info("labels: {}".format(example.labels))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            self.examples[i].labels
        )


def eval_metrics_per_class(
        run, npt_logger, log_neptune, namespace, accs, precs, recs, f1s, f1_macros, mccs
):
    target_cols = [
        "access_vector", "access_complexity", "authentication",
        "confidentiality", "integrity", "availability", "severity"
    ]

    if run and npt_logger and log_neptune:
        for i, col in enumerate(target_cols):
            run[npt_logger.base_namespace][f"{namespace}/{col}/acc_metric"].append(accs[i])
            run[npt_logger.base_namespace][f"{namespace}/{col}/precision"].append(precs[i])
            run[npt_logger.base_namespace][f"{namespace}/{col}/recall"].append(recs[i])
            run[npt_logger.base_namespace][f"{namespace}/{col}/f_measure"].append(f1s[i])
            run[npt_logger.base_namespace][f"{namespace}/{col}/macro_f_measure"].append(f1_macros[i])
            run[npt_logger.base_namespace][f"{namespace}/{col}/mcc"].append(mccs[i])


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(tune_params, config, args, train_dataset, model, tokenizer,
          run=None, npt_logger=None, log_neptune=False, ):
    """ Train the model """

    if args.model == "devign":
        model = DevignModel(tune_params, model, config, tokenizer, args)
    else:
        model = GNNReGVD(tune_params, model, config, tokenizer, args)
    print(model)
    model.to(args.device)

    if log_neptune:
        run['args'] = args
        npt_logger = NeptuneLogger(
            run=run,
            model=model,
            log_model_diagram=True,
            log_gradients=True,
            log_parameters=True,
            log_freq=30,
        )
        run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(
            args
        )

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Data loader len {len(train_dataloader)}")
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=tune_params["learning_rate"], eps=tune_params["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.max_steps * 0.1,
        num_training_steps=args.max_steps
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0

    metrics_es = f"eval_{args.early_stopping_metric}"
    best_metrics_es = -1.0
    model.zero_grad()

    early_stopping_count = 0
    max_early_stopping_count_epoch = args.max_early_stopping

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        logger.info(f"Starting epoch {idx}")
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0:
                logger.info(f"      Starting batch {step} processing")
            inputs = batch[0].to(args.device)
            # labels = batch[1].to(args.device)
            # change to a list of labels
            labels_list = [label.to(args.device) for label in batch[1]]

            model.train()
            loss, logits = model(inputs, labels_list)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if global_step % 100 == 0:
                    logger.info(f"      Performing optimization step {global_step}")

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

        avg_loss = round(train_loss / tr_num, 5)
        logger.info(" epoch {} average loss {}".format(idx, avg_loss))

        if run and npt_logger and log_neptune:
            run[npt_logger.base_namespace]["epoch/loss"].append(avg_loss)

        logger.info(f"  Running eval over epoch {idx}:")
        results_epoch = evaluate(
            args, model, tokenizer, True,
            run, npt_logger, log_neptune, "epoch"
        )

        for key, value in results_epoch.items():
            logger.info("  %s = %s", key, round(value, 4))

            # Save model checkpoint

        logger.info("  " + "*" * 20)
        if results_epoch[metrics_es] > best_metrics_es:

            early_stopping_count = 0
            best_metrics_es = results_epoch[metrics_es]
            logger.info(f"  New Best {metrics_es} in epoch {idx}: {best_metrics_es}", )

            checkpoint_prefix = f'checkpoint-best-{args.early_stopping_metric}'
            output_dir_best_metrics_es = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))

            if not os.path.exists(output_dir_best_metrics_es):
                os.makedirs(output_dir_best_metrics_es)

            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir_best_metrics_es = os.path.join(output_dir_best_metrics_es, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir_best_metrics_es)
            logger.info(f"Saving model checkpoint for best {metrics_es} to %s", output_dir_best_metrics_es)

        elif results_epoch[metrics_es] == 0.0:
            logger.info(f"  {metrics_es} is 0.0, keeping previous early stopping counter and continue training."
                        f"  Epoch {idx}, early stopping counter {early_stopping_count}")

        else:
            early_stopping_count += 1
            logger.info(f"  {metrics_es} does not improve over epoch {idx},"
                        f" early stopping counter: {early_stopping_count}")
            logger.info(f"  Current best {metrics_es} over epochs: {best_metrics_es}")

            if early_stopping_count >= max_early_stopping_count_epoch:
                logger.info(f"  EARLY STOPPING TRIGGERED, epoch {idx}, optimization step {global_step}")
                logger.info(f"  Current best {metrics_es} over epochs {best_metrics_es}")
                break
        logger.info("  " + "*" * 20)
        ray_train.report(
            {f"{args.early_stopping_metric}": results_epoch[metrics_es]},
        )
        print("Finished Training")
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = f'checkpoint-best-{args.early_stopping_metric}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, run, npt_logger)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-{args.early_stopping_metric}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        # test_result = test(args, model, tokenizer)
        test_result = test(args, model, tokenizer, run, npt_logger, args.log_neptune)

        logger.info("***** Test results *****")
        for key in sorted(test_result.keys()):
            logger.info("  %s = %s", key, str(round(test_result[key], 4)))


def evaluate(
        args, model, tokenizer,
        eval_when_training=False, run=None, npt_logger=None, log_neptune=False, stage="batch"
):
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = [[] for _ in range(7)]
    labels = [[] for _ in range(7)]

    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        labels_list = [label.to(args.device) for label in batch[1]]
        with torch.no_grad():
            lm_loss, logit = model(inputs, labels_list)
            eval_loss += lm_loss.mean().item()
            for i, (lgt, lbl) in enumerate(zip(logit, labels_list)):
                logits[i].append(lgt.cpu().numpy())
                labels[i].append(lbl.cpu().numpy())
        nb_eval_steps += 1
    logits = [np.concatenate(logit) for logit in logits]
    labels = [np.concatenate(label) for label in labels]

    probs = [scipy.special.softmax(logit, axis=1) for logit in logits]

    # Apply threshold
    preds = [(prob >= np.max(prob, axis=1, keepdims=True)).astype(int) for prob in probs]

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_acc = np.mean([
        metrics.accuracy_score(labels[i], preds[i])
        for i in range(len(logits))
    ])

    accs = [
        metrics.accuracy_score(true, pred)
        for true, pred in zip(labels, preds)
    ]
    eval_acc_metric = np.mean(accs)

    precs = [
        metrics.precision_score(true, pred, average="macro")
        for true, pred in zip(labels, preds)
    ]
    eval_precision = np.mean(precs)

    recs = [
        metrics.recall_score(true, pred, average="macro")
        for true, pred in zip(labels, preds)
    ]
    eval_recall = np.mean(recs)

    f1s = [
        metrics.f1_score(true, pred)
        for true, pred in zip(labels, preds)]
    eval_f1 = np.mean(f1s)

    f1_macros = [
        metrics.f1_score(y_true=true, y_pred=pred, average="macro")
        for true, pred in zip(labels, preds)]
    eval_f1_macro = np.mean(f1_macros)

    mcc_labels = [label.argmax(axis=1) for label in labels]
    mcc_preds = [pred.argmax(axis=1) for pred in preds]

    mccs = [
        metrics.matthews_corrcoef(y_true=true, y_pred=pred)
        for true, pred in zip(mcc_labels, mcc_preds)
    ]
    eval_mcc = np.mean(mccs)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_acc_metric": eval_acc_metric,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "eval_f1_macro": eval_f1_macro,
        "eval_mcc": eval_mcc,
    }

    if run and npt_logger and log_neptune:
        if eval_when_training:
            run[npt_logger.base_namespace][f"train/{stage}/loss"].append(float(perplexity))
            run[npt_logger.base_namespace][f"train/{stage}/acc"].append(round(eval_acc, 4))
            run[npt_logger.base_namespace][f"train/{stage}/acc_metric"].append(eval_acc_metric)
            run[npt_logger.base_namespace][f"train/{stage}/precision"].append(eval_precision)
            run[npt_logger.base_namespace][f"train/{stage}/recall"].append(eval_recall)
            run[npt_logger.base_namespace][f"train/{stage}/f_measure"].append(eval_f1)
            run[npt_logger.base_namespace][f"train/{stage}/macro_f_measure"].append(eval_f1_macro)
            run[npt_logger.base_namespace][f"train/{stage}/mcc"].append(eval_mcc)
            eval_metrics_per_class(
                run, npt_logger, log_neptune, "train",
                accs, precs, recs, f1s, f1_macros, mccs
            )
        else:
            run[npt_logger.base_namespace][f"eval/{stage}/loss"].append(float(perplexity))
            run[npt_logger.base_namespace][f"eval/{stage}/acc"].append(round(eval_acc, 4))
            run[npt_logger.base_namespace][f"eval/{stage}/acc_metric"].append(eval_acc_metric)
            run[npt_logger.base_namespace][f"eval/{stage}/precision"].append(eval_precision)
            run[npt_logger.base_namespace][f"eval/{stage}/recall"].append(eval_recall)
            run[npt_logger.base_namespace][f"eval/{stage}/f_measure"].append(eval_f1)
            run[npt_logger.base_namespace][f"eval/{stage}/macro_f_measure"].append(eval_f1_macro)
            eval_metrics_per_class(
                run, npt_logger, log_neptune, "eval",
                accs, precs, recs, f1s, f1_macros, mccs
            )

    target_cols = [
        "access_vector", "access_complexity", "authentication",
        "confidentiality", "integrity", "availability", "severity"
    ]
    # since the order is ensured, we can get the corresponding result by idx
    for i, col in enumerate(target_cols):
        result.update({
            f"eval_acc_metric_{col}": accs[i],
            f"eval_precision_{col}": precs[i],
            f"eval_recall_{col}": recs[i],
            f"eval_f1_{col}": f1s[i],
            f"eval_f1_macro_{col}": f1_macros[i],
            f"eval_mcc_{col}": mccs[i],
        })

    return result


def test(args, model, tokenizer, run=None, npt_logger=None, log_neptune=False):
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    logits = [[] for _ in range(7)]
    labels = [[] for _ in range(7)]

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        labels_list = [label.to(args.device) for label in batch[1]]
        with torch.no_grad():
            lm_loss, logit = model(inputs, labels_list)
            eval_loss += lm_loss.mean().item()
            for i, (lgt, lbl) in enumerate(zip(logit, labels_list)):
                logits[i].append(lgt.cpu().numpy())
                labels[i].append(lbl.cpu().numpy())
        nb_eval_steps += 1
    logits = [np.concatenate(logit) for logit in logits]
    labels = [np.concatenate(label) for label in labels]

    # Convert logits to probabilities
    probs = [scipy.special.softmax(logit, axis=1) for logit in logits]

    # Convert logits to class predictions
    preds = [(prob >= np.max(prob, axis=1, keepdims=True)).astype(int) for prob in probs]

    test_acc = np.mean([
        metrics.accuracy_score(labels[i], preds[i])
        for i in range(len(logits))
    ])

    accs = [
        metrics.accuracy_score(true, pred)
        for true, pred in zip(labels, preds)
    ]
    test_acc_metric = np.mean(accs)

    precs = [
        metrics.precision_score(true, pred, average="macro")
        for true, pred in zip(labels, preds)
    ]
    test_precision = np.mean(precs)

    recs = [
        metrics.recall_score(true, pred, average="macro")
        for true, pred in zip(labels, preds)
    ]
    test_recall = np.mean(recs)

    f1s = [
        metrics.f1_score(true, pred)
        for true, pred in zip(labels, preds)]
    test_f1 = np.mean(f1s)

    f1_macros = [
        metrics.f1_score(y_true=true, y_pred=pred, average="macro")
        for true, pred in zip(labels, preds)
    ]
    test_f1_macro = np.mean(f1_macros)

    mcc_labels = [label.argmax(axis=1) for label in labels]
    mcc_preds = [pred.argmax(axis=1) for pred in preds]

    mccs = [
        metrics.matthews_corrcoef(y_true=true, y_pred=pred)
        for true, pred in zip(mcc_labels, mcc_preds)
    ]
    test_mcc = np.mean(mccs)

    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(
                eval_dataset.examples, zip(*preds)
        ):  # Unpack preds so each pred is a tuple of class predictions
            class_labels = [np.argmax(p) for p in pred]
            f.write(f"{example.idx},{','.join(map(str, class_labels))}\n")  # Write example id and all class predictions

    result = {
        "test_acc": round(test_acc, 4),
        "test_acc_metric": test_acc_metric,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_f1_macro": test_f1_macro,
        "test_mcc": test_mcc
    }

    if run and npt_logger and log_neptune:
        run[npt_logger.base_namespace]["test/acc"] = test_acc
        run[npt_logger.base_namespace]["test/acc_metric"] = test_acc_metric
        run[npt_logger.base_namespace]["test/precision"] = test_precision
        run[npt_logger.base_namespace]["test/recall"] = test_recall
        run[npt_logger.base_namespace]["test/f1"] = test_f1
        run[npt_logger.base_namespace]["test/f1_macro"] = test_f1_macro
        run[npt_logger.base_namespace]["test/mcc"] = test_mcc
        eval_metrics_per_class(
            run, npt_logger, log_neptune, "eval",
            accs, precs, recs, f1s, f1_macros, mccs
        )

    target_cols = [
        "access_vector", "access_complexity", "authentication",
        "confidentiality", "integrity", "availability", "severity"
    ]
    # since the order is ensured, we can get the corresponding result by idx
    for i, col in enumerate(target_cols):
        result.update({
            f"test_acc_metric_{col}": accs[i],
            f"test_precision_{col}": precs[i],
            f"test_recall_{col}": recs[i],
            f"test_f1_{col}": f1s[i],
            f"test_f1_macro_{col}": f1_macros[i],
            f"test_mcc_{col}": mccs[i],
        })

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="../input/train.jsonl", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default="../input/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="../input/test.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    # parser.add_argument("--learning_rate", default=5e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay", default=0.0, type=float,
    #                     help="Weight decay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model", default="GNNs", type=str, help="")
    # parser.add_argument("--hidden_size", default=256, type=int,
    #                     help="hidden size.")
    # parser.add_argument("--feature_dim_size", default=768, type=int,
    #                     help="feature dim size.")
    # parser.add_argument("--num_GNN_layers", default=2, type=int,
    #                     help="num GNN layers.")
    # parser.add_argument("--num_classes", default=2, type=int,
    #                     help="num classes.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN/ReGGNN/Devign")

    parser.add_argument("--format", default="uni", type=str,
                        help="idx for index-focused method, uni for unique token-focused method")
    # parser.add_argument("--window_size", default=3, type=int, help="window_size to build graph")
    # parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    # parser.add_argument("--att_op", default='mul', type=str,
    #                     help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--training_percent", default=1., type=float, help="percent of training sample")
    parser.add_argument('--early_stopping_metric', type=str, default='mcc')
    parser.add_argument('--max_early_stopping', type=int, default=5)
    parser.add_argument("--log_neptune", default=False, action='store_true')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # logs
    run = None
    npt_logger = None

    if args.log_neptune:
        print("custom_run_id", f'm{args.model}_e{args.epoch}_t{hex(int(time.time()))[2:]}')
        run = neptune.init_run(
            name=f"m{args.model}_e{args.epoch}_t{datetime.now().strftime('%d_%m_%Y_%H_%M')}",
            custom_run_id=f"m{args.model}_e{args.epoch}_t{hex(int(time.time()))[2:]}",
            project="",
            api_token="",
        )

    args.device = device
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    else:
        model = model_class(config)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        logger.info("Initializing training dataset")
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)

        tune_params = {
            "learning_rate": tune.loguniform(5e-4, 1e-1),
            # "batch_sizes": tune.choice(tune_batch_sizes),
            # "weight_decays": tune.choice(tune_weight_decays),
            # "dropout_rates": tune.choice(tune_dropout_rates),
            # "feature_dim_size": tune.choice([768, 1024, 2048]),
            "hidden_size": tune.choice([32, 64, 128, 256, 512]),
            "num_GNN_layers": tune.choice([1, 2, 3, 4, 5]),
            "att_op": tune.choice(["mul", "sum", "concat"]),
            "window_size": tune.choice([1, 3, 5, 7, 9]),
            "adam_epsilon": tune.loguniform(1e-8, 1e-4),
        }

        scheduler = ASHAScheduler(
            max_t=args.epoch,
            grace_period=1,
            reduction_factor=2)
        os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    train,
                    config=config,
                    args=args,
                    train_dataset=train_dataset,
                    model=model,
                    tokenizer=tokenizer,
                    # run=run,
                    # npt_logger=npt_logger,
                    log_neptune=args.log_neptune
                ),
                resources={"cpu": 16, "gpu": 1}
            ),
            tune_config=tune.TuneConfig(
                metric=args.early_stopping_metric,
                mode="max",
                scheduler=scheduler,
                num_samples=10,
                max_concurrent_trials=1
            ),
            param_space=tune_params,
        )
        results = tuner.fit()
        best_result = results.get_best_result(args.early_stopping_metric, "max")

        print("Best trial config: {}".format(best_result.config))
        print(f"Best trial final validation {args.early_stopping_metric}:"
              f" {best_result.metrics[args.early_stopping_metric]}")


if __name__ == "__main__":
    main()
