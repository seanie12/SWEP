import os
import argparse

import torch
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer)

from models import VariationalBert, VariationalElectra
from mrqa_utils import evaluate, read_answers, read_predictions
from run_squad import get_dataloader, to_list

from squad_metrics import (SquadResult, compute_predictions_logits,
                           squad_evaluate)
from squad_utils import (convert_examples_to_features, read_mrqa_examples,
                         read_squad_examples)


def run(args):
    ckpt = torch.load(args.ckpt_file, map_location="cpu")
    state_dict = ckpt["state_dict"]
    model_args = ckpt["args"]
    device = torch.cuda.current_device()
    
    
    if args.baseline:
        model = AutoModelForQuestionAnswering.from_pretrained(model_args.bert_model)
    else:
        if model_args.electra:
            model = VariationalElectra(model_args)
        else:    
            model = VariationalBert(model_args)
    model.load_state_dict(state_dict)

    model.eval()
    model = model.to(device)
    prefix = ""
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    examples = read_mrqa_examples(args.test_file,
                                  debug=args.debug,
                                  is_training=False)
    features = convert_examples_to_features(examples, tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            max_query_length=args.max_query_length,
                                            doc_stride=args.doc_stride,
                                            is_training=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_dataloader = get_dataloader(args, features, is_training=False)

    # Eval!
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        input_ids, mask, seg_ids, feature_indices = batch

        length = torch.sum(mask, 1)
        max_length = torch.max(length)

        input_ids = input_ids[:, :max_length].to(device)
        mask = mask[:, :max_length].to(device)
        seg_ids = seg_ids[:, :max_length].to(device)

        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "token_type_ids": seg_ids,
            }

            outputs = model(**inputs)
            outputs = (outputs[0], outputs[1])

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

    answers = read_answers(args.test_file)
    metrics = evaluate(answers, predictions, args.skip_no_answer)
    eval_file = os.path.join(args.output_dir, "bioasq_metrics.txt")
    with open(eval_file, "w") as f:
        for k, v in metrics.items():
            str_format = "{}: {:.4f}".format(k, v)
            print(str_format)
            f.write(str_format + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_file", required=True, type=str)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str,
                        default="./mrqa-data/BioASQ.jsonl.gz")
    parser.add_argument("--skip_no_answer", action="store_true")

    parser.add_argument("--version_2_with_negative", action="store_true")
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--verbose_logging", action="store_true")
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)

    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--baseline", action="store_true")

    parser.add_argument("--var", action="store_true")
    
    args = parser.parse_args()
    args.do_lower_case = True
    
    run(args)
