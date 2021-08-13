import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from models import  VariationalBert, VariationalElectra
from run_squad import get_dataloader, to_list
from squad_metrics import (SquadResult, compute_predictions_logits,
                           squad_evaluate)
from squad_utils import convert_examples_to_features, read_squad_examples


def run(args):
    ckpt = torch.load(args.ckpt_file, map_location="cpu")
    state_dict = ckpt["state_dict"]
    model_args = ckpt["args"]
    
    if args.baseline:
        model = AutoModelForQuestionAnswering.from_pretrained(model_args.bert_model)
    else:
        if model_args.electra:
            model = VariationalElectra(model_args)
        else:    
            model = VariationalBert(model_args)
    
    model.load_state_dict(state_dict)
    device = torch.cuda.current_device()
    args.device = device
    
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_args.bert_model)
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

    prefix = ""
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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(args.output_dir, "{}_metrics.txt".format(args.data))
    f = open(output_file, "w")
    for k,v in results.items():
        print("{}: {:.4f}".format(k,v ))
        f.write("{}: {:.4f}\n".format(k, v))
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_folder", type=str, default="shift-data")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--max_query_length", type=int, default=64)

    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1004)

    parser.add_argument("--version_2_with_negative", action="store_true")
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--verbose_logging", action="store_true")
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    
    args.do_lower_case = True
    for file_name in os.listdir(args.test_folder):
        args.data = file_name.replace("_v1.0.json","")
        args.dev_file = os.path.join(args.test_folder, file_name)
        run(args)
