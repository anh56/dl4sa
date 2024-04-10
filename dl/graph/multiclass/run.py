from __future__ import absolute_import, division, print_function

from datetime import datetime
import time
import argparse
import logging
import os
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import json
from sklearn import metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import multiprocessing
from model import *
import ray
from ray import tune
from ray import train as ray_train
from ray.tune.schedulers import ASHAScheduler

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

# enable if
ray.init(
    num_gpus=1,
    # dashboard_port=8265,
    # dashboard_host="0.0.0.0",
    # include_dashboard=True
)


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


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


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['idx'], js[args.target_class])


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
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].label)
        )


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
        print("Using devign")
        model = DevignModel(tune_params, model, config, tokenizer, args)
    else:  # GNNs
        model = GNNReGVD(tune_params, model, config, tokenizer, args)

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

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
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

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0

    metrics_es = f"eval_{args.early_stopping_metric}"
    best_metrics_es = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    early_stopping_count = 0
    max_early_stopping_count_epoch = args.max_early_stopping
    # max_early_stopping_count_batch = 100

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        logger.info(f"Starting epoch {idx}")
        # early_stopping_count_batch = 0

        # bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        # for step, batch in enumerate(bar):
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0:
                logger.info(f"      Starting batch {step} processing")
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)

            # bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            # logger.info("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if global_step % 100 == 0:
                    logger.info(f"      Performing optimization step {global_step}")

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True

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
        checkpoint_prefix = f'checkpoint-best-{args.early_stopping_metric}/model.bin'
        ray_train.report(
            {f"{args.early_stopping_metric}": results_epoch[metrics_es]},
        )
        print("Finished Training")
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = f'checkpoint-best-{args.early_stopping_metric}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, run, npt_logger)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = f'checkpoint-best-{args.early_stopping_metric}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        # test_result = test(args, model, tokenizer)
        test_result = test(args, model, tokenizer, run, npt_logger, args.log_neptune)

        logger.info("***** Test results *****")
        for key in sorted(test_result.keys()):
            logger.info("  %s = %s", key, str(round(test_result[key], 4)))


def evaluate(args, model, tokenizer, eval_when_training=False, run=None, npt_logger=None, log_neptune=False,
             stage="batch"):
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []

    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)

    if args.num_classes == 2:
        preds = logits[:, 0] > 0.5
        preds_for_metrics = list(map(int, list(preds)))

        eval_acc = np.mean(labels == preds)
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.tensor(eval_loss)

        eval_acc_metric = metrics.accuracy_score(labels, preds_for_metrics)
        eval_precision = metrics.precision_score(labels, preds_for_metrics)
        eval_recall = metrics.recall_score(labels, preds_for_metrics)
        eval_f1 = metrics.f1_score(labels, preds_for_metrics)
        eval_f1_macro = eval_f1

    else:
        # Choose the class with the highest probability as prediction
        preds = np.argmax(logits, axis=1)
        preds_for_metrics = preds.tolist()

        eval_acc = np.mean(labels == preds)
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.tensor(eval_loss)

        eval_acc_metric = metrics.accuracy_score(labels, preds_for_metrics)
        eval_precision = metrics.precision_score(labels, preds_for_metrics, average="macro")
        eval_recall = metrics.recall_score(labels, preds_for_metrics, average="macro")
        eval_f1 = metrics.f1_score(labels, preds_for_metrics, average="macro")
        eval_f1_macro = eval_f1

    eval_mcc = metrics.matthews_corrcoef(labels, preds_for_metrics)
    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_acc_metric": eval_acc_metric,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "eval_macro_f1": eval_f1,
        "eval_mcc": eval_mcc
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
        else:
            run[npt_logger.base_namespace][f"eval/{stage}/loss"].append(float(perplexity))
            run[npt_logger.base_namespace][f"eval/{stage}/acc"].append(round(eval_acc, 4))
            run[npt_logger.base_namespace][f"eval/{stage}/acc_metric"].append(eval_acc_metric)
            run[npt_logger.base_namespace][f"eval/{stage}/precision"].append(eval_precision)
            run[npt_logger.base_namespace][f"eval/{stage}/recall"].append(eval_recall)
            run[npt_logger.base_namespace][f"eval/{stage}/f_measure"].append(eval_f1)
            run[npt_logger.base_namespace][f"eval/{stage}/macro_f_measure"].append(eval_f1_macro)
    return result


def test(args, model, tokenizer, run=None, npt_logger=None, log_neptune=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    # for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)

    if args.num_classes == 2:
        preds = logits[:, 0] > 0.5
        preds_for_metrics = list(map(int, list(preds)))

        test_acc = np.mean(labels == preds)
        # test_loss = eval_loss / nb_eval_steps
        # perplexity = torch.tensor(eval_loss)
        test_acc_metric = metrics.accuracy_score(labels, preds_for_metrics)
        test_precision = metrics.precision_score(labels, preds_for_metrics)
        test_recall = metrics.recall_score(labels, preds_for_metrics)
        test_f1 = metrics.f1_score(labels, preds_for_metrics)
        test_f1_macro = test_f1
    else:
        # Choose the class with the highest probability as prediction
        preds = np.argmax(logits, axis=1)
        preds_for_metrics = preds.tolist()
        test_acc = np.mean(labels == preds)
        test_acc_metric = metrics.accuracy_score(labels, preds_for_metrics)
        test_precision = metrics.precision_score(labels, preds_for_metrics, average="macro")
        test_recall = metrics.recall_score(labels, preds_for_metrics, average="macro")
        test_f1 = metrics.f1_score(labels, preds_for_metrics, average="macro")
        test_f1_macro = test_f1
    test_mcc = metrics.matthews_corrcoef(labels, preds_for_metrics)

    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            f.write(example.idx + f'\t{pred}\n')

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
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--learning_rate", default=5e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=1,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--model", default="GNNs", type=str, help="")
    # parser.add_argument("--hidden_size", default=256, type=int,
    #                     help="hidden size.")
    # parser.add_argument("--feature_dim_size", default=768, type=int,
    #                     help="feature dim size.")
    # parser.add_argument("--num_GNN_layers", default=2, type=int,
    #                     help="num GNN layers.")
    parser.add_argument("--num_classes", type=int, help="num classes.")
    parser.add_argument("--target_class", type=str, help="target of classification.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN/ReGGNN/Devign")

    parser.add_argument("--format", default="uni", type=str,
                        help="idx for index-focused method, uni for unique token-focused method")
    parser.add_argument("--window_size", default=3, type=int, help="window_size to build graph")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--training_percent", default=1., type=float, help="percent of training sample")
    parser.add_argument("--alpha_weight", default=1., type=float, help="percent of training sample")
    parser.add_argument("--log_neptune", default=False, action='store_true')
    parser.add_argument('--early_stopping_metric', type=str, default='macro_f1')
    parser.add_argument('--max_early_stopping', type=int, default=5)

    args = parser.parse_args()

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

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // max(args.n_gpu, 1)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(args.n_gpu, 1)
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16
    )

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

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

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        logger.info("Initializing training dataset")
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)
        if args.local_rank == 0:
            torch.distributed.barrier()

        # train(args, train_dataset, model, tokenizer)
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
