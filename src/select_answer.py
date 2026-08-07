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

"""Model-based answer selection over BATS sampling trajectories.

This is the second stage of the BATS evaluation protocol. A BATS run produces
multiple verified answers per question (`all_sampling_stat`); after `--do_eval`
scores each question's final answer (`*_evalv1.jsonl`), this script:

  1. scores every intermediate answer in `all_sampling_stat` (appending the
     final-round answer as a candidate when it is not already recorded),
  2. asks a judge model (default gemini/gemini-2.5-flash) to pick the best answer from the
     lettered candidates and their self-verifications (\\boxed{X} output),
  3. reports three metrics:
       pass acc    — any candidate is correct (upper bound)
       select acc  — the judge-selected answer is correct (headline metric)
       first acc   — the first candidate is correct

The judge call is skipped when no candidate is correct (select = 0) or all
candidates are correct (select = 1); it only runs when the choice matters.

Usage:
  python src/select_answer.py -d outputs/agent_bats/<model>/<tag>/<run>_evalv1.jsonl

Output is written next to the input as `*_select.jsonl`, with `eval/score`
replaced by the selection score so downstream analysis uses it.
"""
import argparse
import asyncio
import concurrent.futures
import copy
import itertools
import json
import multiprocessing as mp
import os
import random
import re
import time
import uuid
from typing import Dict, List

from prompts_bats import *
from helpers import extract_verify, score_from_judge_response
from llm import llm_text_gen, check_credentials
from evaluate import JUDGE_MODEL

MAX_CANDIDATES = 10  # cap on answers offered to the judge


def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(data)} entries from {file_path}")
    return data


def extract_last_round(text: str):
    """Extract the last round number from [[Round N]] markers in a trajectory."""
    if not isinstance(text, str):
        return None
    matches = re.findall(r"\[\[Round\s+(\d+)\]\]", text)
    if matches:
        return int(matches[-1])
    return None


def extract_boxed_answer(text: str):
    """Extracts the content of the first LaTeX \\boxed{} command in a string."""
    match = re.search(r"\\boxed\{(.*?)\}", text)
    if match:
        return match.group(1).strip()
    return None


def evaluate_one_sampling(question, gold_ans, pred_ans):
    """Score a single candidate answer against the gold answer (LLM judge)."""
    if gold_ans and pred_ans:
        if gold_ans.strip() == pred_ans.strip():
            return 1, "Direct match"

    prompt = pr_eval_answer_gemini_instruction().strip()
    prompt = prompt.format(
        question=question,
        response=pred_ans,
        correct_answer=gold_ans
    )

    response, _ = llm_text_gen(prompt, model=JUDGE_MODEL)
    if response is None:
        print("Failed to get a response from the model during evaluation.")
        eval_response = ""
    else:
        eval_response = response.text.strip()

    return score_from_judge_response(eval_response), eval_response


async def select_one_async(entry, model):
    """Score all sampled answers for one question, then judge-select the best."""
    new_entry = copy.deepcopy(entry)

    question = entry['question'].strip()
    gold_answer = entry['gold_answer'].strip()
    all_sampling_stat = entry.get("all_sampling_stat", [])

    # Append the final-round answer as a candidate if it is not already recorded
    round_cnt = extract_last_round(entry.get("full_trajectory", ""))
    if not all_sampling_stat or (round_cnt or 0) > all_sampling_stat[-1].get("round", 0):
        tmp_pred = entry.get("predicted_answer", "")
        tmp_verify = extract_verify(entry.get("full_trajectory", ""))
        all_sampling_stat.append({
            "answer": tmp_pred,
            "round": round_cnt,
            "query_count": entry.get("query_count", 0),
            "url_count": entry.get("url_count", 0),
            "trajectory_len": len(entry.get("full_trajectory", "")),
            "predicted_answer": tmp_pred,
            "verify": tmp_verify,
            "verify_decision": entry.get("verify_decision", "NA"),
            "eval/score": entry.get("eval/score", -1),
            "eval/response": entry.get("eval/response", "")
        })

    all_sampling_stat = all_sampling_stat[:MAX_CANDIDATES]

    # Score each candidate that has not been scored yet
    new_all_sampling_stat = []
    for one_stat in all_sampling_stat:
        if one_stat.get("eval/score", -1) < 0:
            score, eval_response = evaluate_one_sampling(
                question, gold_ans=gold_answer, pred_ans=one_stat.get("answer", "")
            )
            one_stat["eval/score"] = score
            one_stat["eval/response"] = eval_response
        new_all_sampling_stat.append(one_stat)
    new_entry["all_sampling_stat"] = new_all_sampling_stat

    scores = [d["eval/score"] for d in new_all_sampling_stat]
    pass_acc = 1 if sum(1 for s in scores if s > 0) > 0 else 0

    preds = [one_d['answer'] for one_d in new_all_sampling_stat]
    verifications = [one_d['verify'] for one_d in new_all_sampling_stat]

    prompt = f"Question: {question}\n\n"
    for pred_i, one_pred in enumerate(preds):
        prompt += f"\nAnswer {chr(65 + pred_i)}: {one_pred}\nVerification: {verifications[pred_i]}\n"

    eval_response = ""
    if pass_acc == 0:
        print(f"0️⃣ Sample has no correct answers in all predictions, skip.")
        select_ans_id = -1
        select_acc = 0

    elif sum(1 for s in scores if s > 0) == len(scores):
        print(f"✅ Sample all predictions are correct, skip.")
        select_ans_id = 0
        select_acc = 1

    else:
        # answer selection by the judge model
        response, _ = llm_text_gen(
            prompt,
            model=model,
            si=pr_system_instruction_answer_selection(),
            max_output_tokens=30000,
            temperature=0.7,
            seed=42,
        )
        try:
            eval_response = response.text.strip()
            select_ans = extract_boxed_answer(eval_response).strip().upper() if extract_boxed_answer(eval_response) else ""
            select_ans_id = ord(select_ans) - ord('A')
        except Exception as e:
            print(f"Error in extracting boxed answer: {e}")
            eval_response = ""
            select_ans_id = random.randint(0, len(preds) - 1)

        select_acc = new_all_sampling_stat[select_ans_id]['eval/score'] if 0 <= select_ans_id < len(preds) else 0.0

    print(f"✅ Select answer id {select_ans_id} with score {select_acc}, pass score {pass_acc}.\nGold answer: {entry['gold_answer']}")

    new_entry["select/id"] = select_ans_id
    new_entry["select/score"] = select_acc
    new_entry["select/pass_score"] = pass_acc
    new_entry["select/response"] = eval_response.strip()
    new_entry["eval/score"] = select_acc

    return new_entry


async def run_workflow_for_chunk(data_entries, model, temp_file_path):
    tasks = [select_one_async(entry, model) for entry in data_entries]

    with open(temp_file_path, 'w') as f:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                f.write(json.dumps(result) + '\n')
                f.flush()
            except Exception as e:
                print(f"[PID:{os.getpid()}] ❌ Error in task: {e}", flush=True)

    print(f"[PID:{os.getpid()}] 🚀 Finished chunk. All results saved to {temp_file_path}.", flush=True)


def process_chunk_entry_point(chunk_and_args):
    data_chunk, model, temp_dir = chunk_and_args
    temp_file_path = os.path.join(temp_dir, f"temp_results_{os.getpid()}_{uuid.uuid4()}.jsonl")
    print(f"[PID:{os.getpid()}] Assigned chunk of {len(data_chunk)} entries. Writing to {temp_file_path}", flush=True)

    asyncio.run(run_workflow_for_chunk(data_chunk, model, temp_file_path))
    return temp_file_path


def select_main(data_path, save_path, model="gemini-2.5-flash", concurrency=20):
    merge_data = load_jsonl(data_path)

    output_dir = os.path.dirname(save_path)
    temp_dir = os.path.join(output_dir, f"temp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)

    start_time = time.time()

    print(f"🏃 Running multi-process on {len(merge_data)} data with concurrency {concurrency} ...", flush=True)
    num_processes = concurrency
    n_entries = len(merge_data)
    chunk_size = (n_entries + num_processes - 1) // num_processes
    data_chunks = [
        merge_data[i : i + chunk_size]
        for i in range(0, n_entries, chunk_size)
    ]

    temp_file_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        chunk_and_args_iterator = zip(
            data_chunks,
            itertools.repeat(model),
            itertools.repeat(temp_dir),
        )
        temp_file_paths = list(executor.map(process_chunk_entry_point, chunk_and_args_iterator))

    print("\n\n--- ✅ All processes finished. Merging results... ---", flush=True)
    all_results = []
    for temp_file in temp_file_paths:
        try:
            with open(temp_file, 'r') as f:
                for line in f:
                    all_results.append(json.loads(line))
            os.remove(temp_file)
        except FileNotFoundError:
            print(f"Warning: Temp file {temp_file} not found.", flush=True)

    all_results.sort(key=lambda x: x['index'])

    with open(save_path, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')

    try:
        os.rmdir(temp_dir)
    except OSError as e:
        print(f"Warning: Could not remove temp directory {temp_dir}. Error: {e}", flush=True)

    exe_time = time.time() - start_time
    print(f"\n🎉 Successfully saved {len(all_results)} results to {save_path}", flush=True)
    print(f"⏱️ Total time taken: {exe_time // 60:.0f}m, {exe_time % 60:.2f}s", flush=True)


def get_first_score(data):
    score = 0
    for inst in data:
        one_score = inst['all_sampling_stat'][0].get('eval/score', 0)
        score += one_score
    return score / len(data) if data else 0.0


def report_scores(output_path):
    output_data = load_jsonl(output_path)
    all_pass_acc = [inst['select/pass_score'] for inst in output_data]
    all_select_acc = [inst['select/score'] for inst in output_data]

    pass_acc = sum(all_pass_acc) / len(all_pass_acc) if all_pass_acc else 0.0
    select_acc = sum(all_select_acc) / len(all_select_acc) if all_select_acc else 0.0
    first_acc = get_first_score(output_data)
    print(f"pass acc: {pass_acc:.4f}, select acc: {select_acc:.4f}, first ans acc: {first_acc:.4f}")


def analyze_pass_select(output_path):
    """List questions where a correct answer existed but the judge picked a wrong one."""
    output_data = load_jsonl(output_path)
    ids = []
    for i, inst in enumerate(output_data):
        if inst['select/pass_score'] > 0 and inst['select/score'] == 0:
            ids.append(i)
            print(f"Pass but select wrong: {i}, gold: {inst['gold_answer']}, "
                  f"all preds: {[d['answer'] for d in inst['all_sampling_stat']]}, select id: {inst['select/id']}")
    print(f"ids: {ids}")


def main():
    parser = argparse.ArgumentParser(description="Model-based answer selection over BATS sampling trajectories.")
    parser.add_argument("-d", "--data_path", type=str, required=True,
                        help="Path to a *_evalv1.jsonl results file (output of a run with --do_eval).")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash",
                        help="Judge model for answer selection (LiteLLM name, e.g. gemini/gemini-2.5-flash).")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of parallel processes.")
    parser.add_argument("--analyze", action='store_true',
                        help="Also list questions where a correct candidate existed but selection failed.")
    args = parser.parse_args()
    check_credentials(args.model)
    check_credentials(JUDGE_MODEL)

    output_path = args.data_path.replace(".jsonl", "_select.jsonl")
    if os.path.exists(output_path):
        print(f"Selection results already exist at {output_path}. Skipping selection.")
        print(f"If you want to re-run, please delete the existing file:\nrm {output_path}")
    else:
        select_main(args.data_path, output_path, model=args.model, concurrency=args.concurrency)

    report_scores(output_path)
    if args.analyze:
        analyze_pass_select(output_path)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
