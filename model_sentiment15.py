import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Dict

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--train_path", type=str, default="sentiment15-preprocessed/train.csv")
parser.add_argument("--dev_path", type=str, default="sentiment15-preprocessed/dev.csv")
parser.add_argument("--test_path", type=str, default="sentiment15-preprocessed/test.csv")

parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/crosloengual-bert")
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--eval_every_n_examples", type=int, default=20000)


if __name__ == "__main__":
    args = parser.parse_args()
    RANDOM_SEED = 17

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.makedirs(args.experiment_dir, exist_ok=True)
    ts = time.time()
    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, f"train{ts}.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    if args.max_length is None:
        args.max_length = 64
        logging.warning("--max_length is not set. Using max_length=64 as default, but you should set this to a "
                        "reasonable number such as the 95th or 99th percentile of training sequence lengths")

    with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), fp=f, indent=4)

    EVAL_EVERY_N_BATCHES = (args.eval_every_n_examples + args.batch_size - 1) // args.batch_size

    data = datasets.load_dataset("csv", data_files={
        "train": args.train_path,
        "validation": args.dev_path,
        "test": args.test_path
    })

    # Mark the target feature
    data = data.class_encode_column("sentiment")
    data = data.rename_column("sentiment", "labels")  # most models expect targets inside `labels=...` argument
    num_ex = len(data["train"]) + len(data["validation"]) + len(data["test"])

    # TODO: check that this mapping is saved when the model checkpoint is saved later
    id2label = dict(enumerate(data["train"].features["labels"].names))
    label2id = {_lbl: _i for _i, _lbl in id2label.items()}
    num_classes = len(id2label)

    logging.info(f"Loaded {num_ex} examples: "
                 f"{len(data['train'])} train, "
                 f"{len(data['validation'])} dev, "
                 f"{len(data['test'])} test")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[MENTION]"]})

    # TODO: uncomment this to find a reasonable max length to pad sequences to
    # tmp_encoded = tokenizer.batch_encode_plus(data["train"]["content"])
    # tmp_lengths = sorted([len(_curr) for _curr in tmp_encoded["input_ids"]])
    # print(f"95th perc.: {tmp_lengths[int(0.95 * len(tmp_lengths))]}")
    # print(f"99th perc.: {tmp_lengths[int(0.99 * len(tmp_lengths))]}")
    # exit(0)
    # TODO: ----

    def tokenize_function(examples):
        return tokenizer(examples["content"], padding="max_length", max_length=args.max_length, truncation=True)

    tokenized_data = data.map(tokenize_function, batched=True)
    train_data = tokenized_data["train"]
    dev_data = tokenized_data["validation"]
    test_data = tokenized_data["test"]

    train_distribution = {id2label[_lbl_int]: _count / len(train_data)
                          for _lbl_int, _count in Counter(train_data['labels']).most_common()}
    logging.info(f"Training distribution: {train_distribution}")

    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_name_or_path,
                                                               num_labels=num_classes,
                                                               id2label=id2label, label2id=label2id)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.experiment_dir,
        do_train=True, do_eval=True, do_predict=True,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_strategy="steps", logging_steps=EVAL_EVERY_N_BATCHES,
        save_strategy="steps", save_steps=EVAL_EVERY_N_BATCHES, save_total_limit=1,
        seed=RANDOM_SEED, data_seed=RANDOM_SEED,
        evaluation_strategy="steps", eval_steps=EVAL_EVERY_N_BATCHES,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        optim="adamw_torch",
        report_to="none",
    )
    accuracy_func = evaluate.load("accuracy")
    precision_func = evaluate.load("precision")
    recall_func = evaluate.load("recall")
    f1_func = evaluate.load("f1")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, int]:
        pred_logits, ground_truth = eval_pred
        predictions = np.argmax(pred_logits, axis=-1)

        metrics = {}
        metrics.update(accuracy_func.compute(predictions=predictions, references=ground_truth))
        macro_f1 = 0.0
        for _lbl_int, _lbl_str in id2label.items():
            bin_preds = (predictions == _lbl_int).astype(np.int32)
            bin_ground_truth = (ground_truth == _lbl_int).astype(np.int32)

            curr = precision_func.compute(predictions=bin_preds, references=bin_ground_truth)
            curr[f"precision_{_lbl_str}"] = curr.pop("precision")
            curr.update(recall_func.compute(predictions=bin_preds, references=bin_ground_truth))
            curr[f"recall_{_lbl_str}"] = curr.pop("recall")
            curr.update(f1_func.compute(predictions=bin_preds, references=bin_ground_truth))
            curr[f"f1_{_lbl_str}"] = curr.pop("f1")
            macro_f1 += curr[f"f1_{_lbl_str}"]

            metrics.update(curr)

        metrics["f1_macro"] = macro_f1 / max(1, len(id2label))

        return metrics

    trainer = Trainer(
        model=model, args=training_args, tokenizer=tokenizer,
        train_dataset=train_data, eval_dataset=dev_data,
        compute_metrics=compute_metrics
    )

    train_metrics = trainer.train()

    for split_name, split_data in [("train", train_data), ("validation", dev_data), ("test", test_data)]:
        curr_metrics = trainer.predict(test_dataset=split_data)
        logging.info(f"{split_name.upper()} metrics:\n\t{curr_metrics.metrics}")
        pred_probas = torch.softmax(torch.from_numpy(curr_metrics.predictions), dim=-1)
        pred_class = torch.argmax(pred_probas, dim=-1)
        curr_res = pd.DataFrame({
            "text": data[split_name]["content"],
            "pred_probas": pred_probas.tolist(),
            "pred_class": list(map(lambda _lbl_int: id2label[_lbl_int], pred_class.tolist())),
            "correct_class": list(map(lambda _lbl_int: id2label[_lbl_int], split_data["labels"]))
        })
        curr_res.to_json(os.path.join(args.experiment_dir, f"{split_name}_preds.json"), orient="records", lines=True)














