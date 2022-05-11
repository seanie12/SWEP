import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (AdamW, AutoModelForQuestionAnswering, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from models import VariationalBert, VariationalElectra
from squad_metrics import (SquadResult, compute_predictions_logits,
                           squad_evaluate)
from squad_utils import convert_examples_to_features, read_squad_examples

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_dataloader(args, features, is_training):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    seg_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if is_training:
        start = torch.tensor(
            [f.start_position for f in features], dtype=torch.long)
        end = torch.tensor(
            [f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(input_ids, mask, seg_ids, start, end)

    else:
        indices = torch.arange(input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(input_ids, mask, seg_ids, indices)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=is_training,
                            num_workers=4)

    return dataloader


def process_batch(batch, device):
    input_ids, mask, seg_ids, start, end = batch

    length = torch.sum(mask, 1)
    max_length = torch.max(length)

    input_ids = input_ids[:, :max_length].to(device)
    mask = mask[:, :max_length].to(device)
    seg_ids = seg_ids[:, :max_length].to(device)

    start = start.to(device)
    end = end.to(device)

    inputs = {"input_ids": input_ids,
              "attention_mask": mask,
              "token_type_ids": seg_ids,
              "start_positions": start,
              "end_positions": end,
              }
    return inputs


def evaluate(args, model, tokenizer, prefix=""):
    examples = read_squad_examples(args.dev_file,
                                   is_training=False,
                                   debug=args.debug)

    features = convert_examples_to_features(examples, tokenizer,
                                            args.max_seq_length,
                                            args.doc_stride,
                                            args.max_query_length,
                                            is_training=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_dataloader = get_dataloader(args, features, is_training=False)

    # Eval!
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", position=3, leave=False):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            feature_indices = batch[3]

            outputs = model(**inputs)
            start_logits, end_logits = outputs[0], outputs[1]
            outputs = (start_logits, end_logits)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def save_model(args, model):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    ckpt_file = os.path.join(args.model_dir, "bert_base.pt")
    ckpt = {"args": args, "state_dict": model.state_dict()}
    torch.save(ckpt, ckpt_file)


def run(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    if args.read_data:
        train_examples = read_squad_examples(args.train_file,
                                             is_training=True,
                                             debug=args.debug)

        train_features = convert_examples_to_features(train_examples,
                                                      tokenizer,
                                                      args.max_seq_length,
                                                      args.doc_stride,
                                                      args.max_query_length,
                                                      is_training=True)
        if not os.path.exists(args.pickle_folder):
            os.makedirs(args.pickle_folder)
        pickle_file = os.path.join(args.pickle_folder, "train_features.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(train_features, f)
            print("save pickle file at: {}".format(pickle_file))

    else:
        pickle_file = os.path.join(args.pickle_folder, "train_features.pkl")
        assert os.path.exists(
            pickle_file) == True, "you must create pickle file set option --read_data"

        with open(pickle_file, "rb") as f:
            train_features = pickle.load(f)

    train_loader = get_dataloader(args, train_features, is_training=True)

    device = torch.cuda.current_device()
    args.device = device
    if args.baseline:
        model = AutoModelForQuestionAnswering.from_pretrained(args.bert_model)
    elif args.electra:
        model = VariationalElectra(args)
    else:
        model = VariationalBert(args)
    model = model.to(device)

    t_total = len(train_loader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)
                       and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)
            and p.requires_grad], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    loss_log = tqdm(total=0, bar_format='{desc}', position=1)
    for _ in range(args.num_train_epochs):
        model.train()
        num_batches = len(train_loader)

        for batch in tqdm(train_loader, total=num_batches, position=0, leave=False):
            inputs = process_batch(batch, device)
            outputs = model(**inputs)
            if args.baseline:
                loss = outputs[0]
                loss_str = "NLL: {:.4f}".format(loss.item())
            else:
                nll, kl = outputs[0], outputs[1]
                loss = nll + kl * args.beta
                loss_str = "NLL: {:.4f}, KL: {:.4f}".format(
                    nll.item(), kl.item())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            loss_log.set_description_str(loss_str)

            if args.debug:
                break

        # save model
        save_model(args, model)

    # load the best model from validation-set
    ckpt_file = os.path.join(args.model_dir, "bert_base.pt")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)

    results = evaluate(args, model, tokenizer)
    output_file = os.path.join(args.model_dir, "metrics.txt")
    with open(output_file, "w") as f:
        for k, v in results.items():
            print("{}: {:.4f}".format(k, v))
            f.write("{}: {:.4f}\n".format(k, v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--dev_file", type=str)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--baseline", action="store_true",
                        help="whether use baseline model")
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")

    parser.add_argument("--model_dir", type=str, default="./save/swep")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=1004)

    parser.add_argument("--version_2_with_negative", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--verbose_logging", action="store_true")
    parser.add_argument("--max_answer_length", type=int, default=30)

    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)

    parser.add_argument("--read_data", action="store_true",
                        help="read data from json file")
    parser.add_argument("--pickle_folder", type=str, default="./noans_pickle")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--electra", action="store_true")
    args = parser.parse_args()

    args.do_lower_case = True

    run(args)
