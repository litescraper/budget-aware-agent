# Copyright 2026 The BATS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ast
import time
import copy
import json
import random
import argparse
from datasets import load_dataset
import requests
import pandas
import base64
import hashlib
import re

from prompts_bats import *
from helpers import score_from_judge_response
from llm import llm_text_gen, check_credentials

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LLM judge used to score predicted answers against the gold answer
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini/gemini-2.5-flash")




def get_query(text: str):
    """Extracts the last search query from the model's output."""
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


def evaluate_one_inst(inst):
    prompt = pr_eval_answer_gemini_instruction().strip()
    prompt = prompt.format(
        question=inst['question'],
        response=inst['predicted_answer'],
        correct_answer=inst['gold_answer']
    )
    print(f"\n\n################# [Evaluation] ##################\n{prompt}")

     
    response, _ = llm_text_gen(prompt, model=JUDGE_MODEL)
    if response is None:
        print("Failed to get a response from the model during evaluation.")
        eval_response = ""
    else:
        eval_response = response.text.strip()

    score = score_from_judge_response(eval_response)

    new_inst = copy.deepcopy(inst)
    new_inst['eval/response'] = eval_response
    new_inst['eval/score'] = score

    print(f"Evaluation Response: {eval_response}")
    print(f"Evaluation Score: {score}")
    return new_inst

def do_evaluate(data_path):
    check_credentials(JUDGE_MODEL)

    eval_save_path = data_path.replace(".jsonl", "_evalv1.jsonl")
    if os.path.exists(eval_save_path):
        print(f"Evaluation results already exist at {eval_save_path}. Skipping evaluation.")
        print("If you want to re-evaluate, please delete the existing file.")
        print(f"rm {eval_save_path}")
        return

    eval_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        try:
            eval_data = [json.loads(line) for line in f]
        except json.JSONDecodeError as e:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    parsed = ast.literal_eval(line)
                    eval_data.append(parsed)
                except Exception as e:
                    print(f"[Warning] Line {line_num} parse failed: {e}")
    print(f"Loaded {len(eval_data)} instances for evaluation from {data_path}")

    eval_results = []
    all_score = []
    for i, inst in enumerate(eval_data):
        print(f"\n--- Evaluating Entry {i} ---")
        evaluated_inst = evaluate_one_inst(inst)
        eval_results.append(evaluated_inst)
        all_score.append(evaluated_inst['eval/score'])
    
    print(f"\n\nEvaluation complete. Total instances: {len(eval_results)}")
    print(f"Average Score: {sum(all_score) / len(all_score) if all_score else 0:.2f}, {sum(all_score)} out of {len(all_score)}")

    # Save evaluation results
    
    with open(eval_save_path, 'w', encoding='utf-8') as f:
        for result in eval_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Evaluation complete. Results saved to {eval_save_path}")
    return sum(all_score) / len(all_score) if all_score else 0


def parser():
    """Argument parser for command line execution."""
    parser = argparse.ArgumentParser(description="Run inference on BrowseComp dataset using Gemini.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model to use for inference.")
    parser.add_argument("--budget", type=int, default=10, help="Maximum number of search iterations per question.")
    parser.add_argument("--begin", type=int, default=0, help="Starting index of the questions to process.")
    parser.add_argument("--end", type=int, default=1, help="Ending index of the questions to process (exclusive).")

    # generation config
    parser.add_argument("--max_output_tokens", type=int, default=20000, help="Maximum output tokens for the model.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the model's generation.")

    # eval
    parser.add_argument("--do_eval", action='store_true', help="Whether to evaluate the answers after inference.")
    parser.add_argument("-d", "--eval_data", type=str, default="", help="Path to the evaluation data file (if applicable).")
    return parser


# --- Main Execution ---
if __name__ == "__main__":

    # Parse command line arguments
    args = parser().parse_args()
    # args.eval_data = "outputs/browsecomp_mp/gemini-2.5-flash/easy_v4/t1.0_b0_e100_bgt30_nqr5_stuck92.jsonl"

    if args.eval_data != "":
        print("Evaluating the results from the provided evaluation data file...")
        do_evaluate(args.eval_data)
        exit()
        