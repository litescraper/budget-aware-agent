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

#!/usr/bin/env python3
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

import re
import json
from typing import List, Tuple, Dict
from tqdm import tqdm
from statistics import mean
import os
import numpy as np
import matplotlib.pyplot as plt
from helpers import *


from dotenv import load_dotenv
load_dotenv()

TOOL_CALL_TAG = os.environ.get("TOOL_CALL_TAG", "tool_code")  # Default to 'tool_call' if not set
# --------- Pricing ---------
MODEL_TOKEN_PRICE = {
    "gemini-2.5-flash": [0.3, 2.5, 0.03],  # [input, output] in USD per 1M tokens
    "gemini-2.5-pro": [1.25, 10.0, 0.125],  # [input, output] in USD per 1M tokens
    "deepseek": [0.6, 1.7, 1.7],  # [input, output] in USD per 1M tokens
    "claude-sonnet": [3, 15.0, 0.3],  # [input, output] in USD per 1M tokens
    "claude-haiku": [1.0, 5.0, 0.1],  # [input, output] in USD per 1M tokens
    "unknown": [0.0, 0.0, 0.0],  # fallback: cost reported as 0 for unpriced models
}
MODEL_NAME = "gemini-2.5-flash"  # default model name

# --------- Plotting ---------
def plot_query_bins_with_accuracy(correct_ans_query_num,
                                  wrong_ans_query_num,
                                  bin_size=5,
                                  max_x=50,
                                  save_dir="figs/analysis",
                                  filename="query_num_distribution_with_acc.png"):
    """
    Plot counts (left y-axis) and per-bin accuracy (right y-axis) with fixed x-maximum.
    accuracy = #correct / (#correct + #wrong) per bin.
    """
    if not correct_ans_query_num and not wrong_ans_query_num:
        raise ValueError("Both input lists are empty.")

    arr_c = np.array(correct_ans_query_num, dtype=int) if correct_ans_query_num else np.array([], dtype=int)
    arr_w = np.array(wrong_ans_query_num, dtype=int) if wrong_ans_query_num else np.array([], dtype=int)

    # Determine max_x if not provided
    data_max = int(np.max([arr_c.max() if arr_c.size else 0,
                           arr_w.max() if arr_w.size else 0]))
    if max_x is None:
        max_x = data_max
    max_x = int(max_x)

    # Define bin edges [0, max_x] step=bin_size
    start = 0
    edges_core = np.arange(start, max_x, bin_size)
    bin_edges = np.concatenate([edges_core, np.array([max_x])]) if edges_core.size else np.array([start, max_x])

    # Histogram counts
    correct_counts, _ = np.histogram(arr_c, bins=bin_edges)
    wrong_counts, _   = np.histogram(arr_w, bins=bin_edges)
    totals = correct_counts + wrong_counts

    # Accuracy per bin
    accuracy = np.divide(correct_counts, totals, out=np.zeros_like(correct_counts, dtype=float), where=totals>0)

    # Bin centers for plotting
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left y-axis: counts
    ax1.plot(centers, correct_counts, marker="o", linewidth=1.5, label="Correct count")
    ax1.plot(centers, wrong_counts, marker="o", linewidth=1.5, label="Wrong count")
    ax1.set_xlabel(f"Query number (binned, width={bin_size})")
    ax1.set_ylabel("# data in bin")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_xlim(0, max_x)
    ax1.legend(loc="upper left")

    # Right y-axis: per-bin accuracy bar
    ax2 = ax1.twinx()
    bar_width = bin_size * 0.8
    ax2.bar(centers, accuracy, width=bar_width, alpha=0.35, label="Accuracy per bin")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # Annotate accuracy values
    for x, acc, tot in zip(centers, accuracy, totals):
        if tot > 0:
            ax2.text(x, acc + 0.02, f"{acc:.2f}", ha="center", va="bottom", fontsize=8)

    # Title uses filename, same size as tick labels, wrap if long
    plt.tight_layout()
    tick_fs = ax1.get_xticklabels()[0].get_size() if ax1.get_xticklabels() else plt.rcParams.get("xtick.labelsize", 10)
    st = fig.suptitle(filename, fontsize=tick_fs)
    st.set_wrap(True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

# --------- Parsing helpers ---------
def safe_sum_pairs(arr: List[int]) -> Tuple[int, int]:
    """
    Given a flat list [in1,out1,in2,out2,...], return total_in, total_out
    """
    if not isinstance(arr, list) or not arr:
        return 0, 0
    tot_in, tot_out = 0, 0
    # only take even pairs
    n_pairs = len(arr) // 2
    for i in range(n_pairs):
        try:
            tot_in += int(arr[2*i]   or 0)
            tot_out+= int(arr[2*i+1] or 0)
        except Exception:
            continue
    return tot_in, tot_out

def safe_get_tokens(arr: List[int]) -> Tuple[int, int, int]:
    if len(arr) == 3:
        return arr[0], arr[1], arr[2]
    elif len(arr) == 2:
        return arr[0], arr[1], 0
    else:
        return 0, 0, 0
    

def split_blocks(text: str) -> List[str]:
    """
    Return a list of substrings, each starting with <tool_call> and ending with the nearest </tool_response>. Case-insensitive.
    """
    txt = (text or "").strip()
    # Some logs may have additional prompts above "Question:", here we simply truncate; if not found, keep original text
    if "Here is your question:" in txt:
        txt = txt.split("Here is your question:")[-1]
    if TOOL_CALL_TAG == "tool_call":
        pattern = re.compile(r'(?i)<tool_call>.*?</tool_response>', re.DOTALL)
    elif TOOL_CALL_TAG == "tool_code":
        pattern = re.compile(r'(?i)<tool_code>.*?</tool_response>', re.DOTALL)
    else:
        raise ValueError(f"Unknown TOOL_CALL_TAG: {TOOL_CALL_TAG}")
    matches = pattern.findall(txt)
    return matches

if TOOL_CALL_TAG == "tool_call":
    TOOL_CALL_RE = re.compile(r'(?is).*\<tool_call\>\s*(.*?)\s*\</tool_call\>')
elif TOOL_CALL_TAG == "tool_code":
    TOOL_CALL_RE = re.compile(r'(?is).*\<tool_code\>\s*(.*?)\s*\</tool_code\>')
else:
    raise ValueError(f"Unknown TOOL_CALL_TAG: {TOOL_CALL_TAG}")
TOOL_RESPONSE_RE   = re.compile(r'(?is).*\<tool_response\>\s*(.*?)\s*\</tool_response\>')

SEARCH_NORESULT_STR = "No results found."
BROWSE_NORESULT_STR = "No content."
SEARCH_API_ERROR_STR = "Search failed: 429, message='Too Many Requests'"
SEARCH_API_ERROR_STR2 = "All API keys are currently on cooldown"
SEARCH_API_ERROR_STR3 = "Search failed"
# X = 5
# SEARCH_API_ERROR_STR3_ALL = [SEARCH_API_ERROR_STR3.replace("X", str(i)) for i in range(1, X+1)]
SEARCH_API_ERROR_STR4 = "HTTP error"
SEARCH_API_ERROR_STRINGS = [SEARCH_API_ERROR_STR, SEARCH_API_ERROR_STR2, SEARCH_API_ERROR_STR3, SEARCH_API_ERROR_STR4]

SEARCH_RESULT_RE = re.compile(r'(?i)^\s*(Search results:|Query:)', re.MULTILINE)
BROWSE_RESULT_RE = re.compile(r'(?i)^\s*(---\nBrowse)', re.MULTILINE)

def parse_block(block: str) -> Tuple[str, str]:
    """
    Given a block, return (tool_call, tool_response), preferring the last tags.
    """
    tool_call_match = TOOL_CALL_RE.search(block or "")
    tool_response_match  = TOOL_RESPONSE_RE.search(block or "")
    tool_call = tool_call_match.group(1).strip() if tool_call_match else ''
    tool_response = tool_response_match.group(1).strip() if tool_response_match else ''
    return tool_call, tool_response

def parse_all(text: str) -> List[Dict[str, str]]:
    return [{'tool_call': q, 'tool_response': i}
            for block in split_blocks(text or "")
            for q, i in [parse_block(block)]]

def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(data)} entries from {file_path}")
    return data

def has_results(tool_response: str) -> bool:
    """
    Heuristic: non-empty AND looks like it contains structured search output.
    """
    if not tool_response or not tool_response.strip():
        return False
    s = tool_response.strip()
    return bool(SEARCH_RESULT_RE.search(s) or BROWSE_RESULT_RE.search(s))

def count_quoted_phrases(text: str) -> int:
    return len(re.findall(r'"[^"]+"', text or ""))

# --------- Safe getters / normalizers ---------
def norm_ans(s: str) -> str:
    return ' '.join((s or "").strip().lower().split())

def robust_query_count(inst: Dict, parsed_blocks_len: int) -> int:
    """
    Priority: query_count > len(queries) > parsed search lines
    """
    if isinstance(inst.get("query_count"), int) and inst["query_count"] >= 0:
        return inst["query_count"]
    if isinstance(inst.get("queries"), list):
        return len(inst["queries"])
    return parsed_blocks_len

def extract_budget_from_path(path: str, key: str) -> int:
    """
    Try to extract int after a key (e.g., 'bgts50' or 'bgtb30'); return -1 if not found.
    """
    m = re.search(rf'(?<={re.escape(key)})\d+', path or "")
    return int(m.group()) if m else -1

# --------- Main analysis ---------
def analyze_tool_results(data_path: str, total: int, budget_search: int, budget_browse: int, plot: bool = False) -> Dict[str, float]:

    global MODEL_NAME
    if "flash" in data_path:
        MODEL_NAME = "gemini-2.5-flash"
    elif "pro" in data_path:
        MODEL_NAME = "gemini-2.5-pro"
    elif "deepseek" in data_path:
        MODEL_NAME = "deepseek"
    elif "sonnet" in data_path:
        MODEL_NAME = "claude-sonnet"
    elif "haiku" in data_path:
        MODEL_NAME = "claude-haiku"
    else:
        MODEL_NAME = "unknown"
        print(f"[analyze] No pricing entry matches '{data_path}'; cost will be reported as 0. "
              f"Add your model to MODEL_TOKEN_PRICE in src/analyze.py for cost estimates.")

    alldata = load_jsonl(data_path)
    if total > 0:
        alldata = alldata[:total]

    # analyze search
    all_query_cnt_list = []
    search_no_result_cnt = 0 # sample level
    query_no_result_cnt = 0 # query level
    
    query_parallel_dict = {}
    quote_stat_dict = {}  # {#quotes: [#none, #total]}
    sample_api_error_cnt = 0 # sample level
    query_api_error_cnt = 0 # SEARCH_API_ERROR_STR Search failed: 429, message='Too Many Requests', url=xxx

    # analyze browse
    all_url_cnt_list = []
    browse_no_result_cnt = 0 # sample level
    url_no_result_cnt = 0 # url level

    url_parallel_dict = {}

    # analyze scores
    all_scores = []
    correct_ans_query_num_list = []
    wrong_ans_query_num_list = []
    correct_ans_url_num_list = []
    wrong_ans_url_num_list = []

    # analyze answers
    none_ans_cnt = 0
    non_none_ans_score_list = []

    # analyze tokens
    all_token_stat = [0, 0, 0]
    summary_token_stat = [0, 0, 0]
    browse_token_stat = [0, 0, 0]  # optional
    verify_token_stat = [0, 0, 0]  # optional

    # continued generation
    cnt_continue = 0
    acc_continue = 0.0

    # v13: verify
    cnt_verify = 0
    cnt_verify_success = 0
    acc_verify = 0.0
    acc_verify_success = 0
    verify_decision_keys = ["NA", "SUCCESS", "CONTINUE", "PIVOT", "NONE"]
    verify_decision_score_dict = {key: [] for key in verify_decision_keys}

    # v9: adaptive memory
    use_ad_sum_score = []
    no_use_ad_sum_score = []
    ad_memory_cnt_list = []

    # overfitting analysis
    tool_code_cnt = 0
    
    for inst in tqdm(alldata, desc="Analyzing search results"):
        blocks = parse_all(inst.get("full_trajectory", ""))

        if "<tool_code>" in inst.get("full_trajectory", "") or "</tool_code>" in inst.get("full_trajectory", ""):
            tool_code_cnt += 1

        # count queries
        n_queries = len(inst['queries'])
        all_query_cnt_list.append(n_queries)

        # count urls
        n_urls = len(inst['urls'])
        all_url_cnt_list.append(n_urls)

        # count occurrences of explicit no result strings
        n_search_no_result = sum((b["tool_response"] or "").lower().count(SEARCH_NORESULT_STR.lower()) for b in blocks)
        n_browse_no_result = sum((b["tool_response"] or "").lower().count(BROWSE_NORESULT_STR.lower()) for b in blocks)
        # count both search api error
        

        # The revised one-liner
        n_api_error = sum(
            any(err.lower() in (b["tool_response"] or "").lower() for err in SEARCH_API_ERROR_STRINGS)
            for b in blocks
        )
        # n_api_error = sum((b["tool_response"] or "").lower().count(SEARCH_API_ERROR_STR.lower()) for b in blocks)

        sample_api_error_cnt += 1 if n_api_error > 0 else 0
        search_no_result_cnt += 1 if n_search_no_result > 0 else 0
        browse_no_result_cnt += 1 if n_browse_no_result > 0 else 0

        # score
        score = inst["eval/score"]
        all_scores.append(float(score))

        # answer analysis

        if inst["predicted_answer"]:
            pred_ans = inst["predicted_answer"].lower().strip()
        else:
            pred_ans = "none"
            
        if pred_ans == "none":
            none_ans_cnt += 1
        else:
            non_none_ans_score_list.append(float(score))

        if float(score) == 1.0:
            correct_ans_query_num_list.append(n_queries)
            correct_ans_url_num_list.append(n_urls)
        else:
            wrong_ans_query_num_list.append(n_queries)
            wrong_ans_url_num_list.append(n_urls)

        # adaptive summary
        ad_cnt = int(inst.get("adaptive_summary_count", 0) or 0)
        ad_memory_cnt_list.append(ad_cnt)
        if ad_cnt > 0:
            use_ad_sum_score.append(float(score))
        else:
            no_use_ad_sum_score.append(float(score))

        # continue flag
        if inst.get("flag_continue", False):
            cnt_continue += 1
            acc_continue += float(score)

        # flag_ans_pass_verification
        if inst.get("flag_ans_pass_verification", None):
            cnt_verify += 1
            acc_verify += float(score)
            if inst.get("flag_ans_pass_verification", False):
                cnt_verify_success += 1
                acc_verify_success += float(score)

        # verify_decision
        vkey = inst.get("verify_decision", "NA")
        verify_decision_score_dict[vkey].append(float(score))

        # token stats (all/summary/browse)
        tot_in, tot_out, tot_cache = safe_get_tokens(inst.get("all_token_counts", []))
        sm_in, sm_out, sm_cache   = safe_get_tokens(inst.get("summary_token_counts", []))
        br_in, br_out, br_cache   = safe_get_tokens(inst.get("browse_token_counts", []))
        ve_in, ve_out, ve_cache   = safe_get_tokens(inst.get("verify_token_counts", []))

        all_token_stat[0]     += tot_in
        all_token_stat[1]     += tot_out
        all_token_stat[2]     += tot_cache

        summary_token_stat[0] += sm_in
        summary_token_stat[1] += sm_out
        summary_token_stat[2] += sm_cache
        browse_token_stat[0]  += br_in
        browse_token_stat[1]  += br_out
        browse_token_stat[2]  += br_cache
        verify_token_stat[0]  += ve_in
        verify_token_stat[1]  += ve_out
        verify_token_stat[2]  += ve_cache

        # fine grained query analysis: quotes + parallel queries
        for b in blocks:

            # parse tool json
            try:
                tool_call = json.loads(b["tool_call"]) if b["tool_call"] else {}
            except Exception as e:
                print(f"Error converting tool call: {e}\n{b.get('tool_call', '')}", flush=True)
                tool_call = {}

            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('arguments', {})
            # if tool is search
            if tool_name == "search":
                # count quotes in search query
                one_block_queries = [q.strip() for q in tool_args.get("query", []) if q.strip()]
                for q in one_block_queries:
                    n_quotes = count_quoted_phrases(q)
                    quote_stat_dict.setdefault(n_quotes, [0, 0])
                    quote_stat_dict[n_quotes][1] += 1
                    if not has_results(b["tool_response"]):
                        quote_stat_dict[n_quotes][0] += 1
                query_parallel_dict[len(one_block_queries)] = query_parallel_dict.get(len(one_block_queries), 0) + 1

                one_no_search_result_cnt = b["tool_response"].lower().count(SEARCH_NORESULT_STR.lower())
                query_no_result_cnt += one_no_search_result_cnt

                one_api_error_cnt = b["tool_response"].lower().count(SEARCH_API_ERROR_STR.lower())
                one_api_error_cnt += b["tool_response"].lower().count(SEARCH_API_ERROR_STR2.lower())
                # if any of the error strings appear, count as one error
                # if any(err in b["tool_response"].lower() for err in SEARCH_API_ERROR_STRINGS):
                #     one_api_error_cnt = 1
                # else:
                #     one_api_error_cnt = 0
                query_api_error_cnt += one_api_error_cnt

            elif tool_name == "browse":
                one_block_urls = [u.strip() for u in tool_args.get("urls", []) if u.strip()]
                url_parallel_dict[len(one_block_urls)] = url_parallel_dict.get(len(one_block_urls), 0) + 1

                one_no_browse_result_cnt = b["tool_response"].lower().count(BROWSE_NORESULT_STR.lower())
                url_no_result_cnt += one_no_browse_result_cnt


    # use-all-budget stats (separately for search/browse)
    cnt_use_all_search_bgts = sum(1 for c in all_query_cnt_list if budget_search > 0 and c == budget_search)
    cnt_use_all_browse_bgts = sum(1 for c in all_url_cnt_list if budget_browse > 0 and c == budget_browse)

    assert len(all_query_cnt_list) == len(alldata), "Query count list length mismatch!"
    assert len(all_url_cnt_list) == len(alldata), "URL count list length mismatch!"
    assert len(all_scores) == len(alldata), "Scores list length mismatch!"

    avg_query_cnt = sum(all_query_cnt_list) / len(all_query_cnt_list) if all_query_cnt_list else 0
    avg_url_cnt = sum(all_url_cnt_list) / len(all_url_cnt_list) if all_url_cnt_list else 0

    print("\n===== Score and Answer Analysis =====")
    print(f"Total queries examined:     {len(alldata):,}")
    print(f"Accuracy:                   {sum(all_scores) / len(all_scores):.4f}" if all_scores else "No accuracy scores found")
    print(f"None answers:               {none_ans_cnt:,}, ratio: {none_ans_cnt / len(alldata):.2%}")
    print(f"Non-none answers:           {len(non_none_ans_score_list):,}, avg score: {mean(non_none_ans_score_list) if non_none_ans_score_list else 0:.4f}")
    print("\n===== Analyze search =====")
    print(f"Avg queries per sample:     {avg_query_cnt:.2f}")
    print(f"Use all search budget:      {cnt_use_all_search_bgts:,}, ratio: {cnt_use_all_search_bgts / len(all_query_cnt_list):.2%}")
    print(f"Search no-result (sample-level):  {search_no_result_cnt:,}, ratio: {search_no_result_cnt / len(alldata):.2%}")
    print(f"Search no-result (query-level):   {query_no_result_cnt:,}, ratio: {query_no_result_cnt / max(sum(all_query_cnt_list), 1):.2%}")
    print(f"Search API error (sample-level):  {sample_api_error_cnt:,}, ratio: {sample_api_error_cnt / len(alldata):.2%}")
    print(f"Search API error (query-level):   {query_api_error_cnt:,}, ratio: {query_api_error_cnt / max(sum(all_query_cnt_list), 1):.2%}")
    print(f"Correct answers (queries):  {len(correct_ans_query_num_list):,}, avg: {mean(correct_ans_query_num_list) if correct_ans_query_num_list else 0:.2f}")
    print(f"Wrong answers (queries):    {len(wrong_ans_query_num_list):,}, avg: {mean(wrong_ans_query_num_list) if wrong_ans_query_num_list else 0:.2f}")
    # none result quote queries
    all_quote_none = sum([v[0] for k, v in quote_stat_dict.items() if int(k) > 0])
    all_quote_total = sum([v[1] for k, v in quote_stat_dict.items() if int(k) > 0])
    if all_quote_total > 0:
        print(f"Quote queries:              {len(quote_stat_dict):,}, none-result: {all_quote_none:,}, total: {all_quote_total:,}, ratio: {all_quote_none / all_quote_total:.2%}")
    else:
        print("No quote queries found.")
    # print(f"Quote stats:                {all_quote_none} / {all_quote_total} ({all_quote_none / all_quote_total:.2%}) none-result quotes")


    print("\n===== Analyze browse =====")
    print(f"Avg URLs per sample:        {avg_url_cnt:.2f}")
    print(f"Use all browse budget:      {cnt_use_all_browse_bgts:,}, ratio: {cnt_use_all_browse_bgts / len(all_url_cnt_list):.2%}")
    print(f"Browse no-result (sample-level):  {browse_no_result_cnt:,}, ratio: {browse_no_result_cnt / len(alldata):.2%}")
    if sum(all_url_cnt_list) > 0:
        print(f"Browse no-result (URL-level):   {url_no_result_cnt:,}, ratio: {url_no_result_cnt / sum(all_url_cnt_list):.2%}")
    else:
        print("No URLs found in browse results.")
    # print(f"Browse no-result (URL-level):   {url_no_result_cnt:,}, ratio: {url_no_result_cnt / sum(all_url_cnt_list):.2%}")
    print(f"Correct answers (URLs):     {len(correct_ans_url_num_list):,}, avg: {mean(correct_ans_url_num_list) if len(correct_ans_url_num_list) > 0 else 0:.2f}")
    print(f"Wrong answers (URLs):       {len(wrong_ans_url_num_list):,}, avg: {mean(wrong_ans_url_num_list) if len(wrong_ans_url_num_list) > 0 else 0:.2f}")

    # token stat + price
    avg_input_tokens  = all_token_stat[0] / len(alldata) if alldata else 0
    avg_output_tokens = all_token_stat[1] / len(alldata) if alldata else 0
    avg_cache_tokens  = all_token_stat[2] / len(alldata) if alldata else 0
    model_price = MODEL_TOKEN_PRICE[MODEL_NAME]
    if "_bf" in data_path:
        avg_browse_input_tokens = browse_token_stat[0] / len(alldata) if alldata else 0
        avg_browse_output_tokens= browse_token_stat[1] / len(alldata) if alldata else 0
        avg_browse_cache_tokens = browse_token_stat[2] / len(alldata) if alldata else 0
        avg_browse_input_cost  = avg_browse_input_tokens  / 1_000_000 * MODEL_TOKEN_PRICE["gemini-2.5-flash"][0]
        avg_browse_output_cost = avg_browse_output_tokens / 1_000_000 * MODEL_TOKEN_PRICE["gemini-2.5-flash"][1]
        avg_browse_cache_cost  = avg_browse_cache_tokens  / 1_000_000 * MODEL_TOKEN_PRICE["gemini-2.5-flash"][2]
        avg_input_cost  = (avg_input_tokens - avg_browse_input_tokens)  / 1_000_000 * model_price[0] + avg_browse_input_cost
        avg_output_cost = (avg_output_tokens - avg_browse_output_tokens) / 1_000_000 * model_price[1] + avg_browse_output_cost
        avg_cache_cost  = (avg_cache_tokens - avg_browse_cache_tokens)  / 1_000_000 * model_price[2] + avg_browse_cache_cost
    else:
        avg_input_cost  = avg_input_tokens  / 1_000_000 * model_price[0]
        avg_output_cost = avg_output_tokens / 1_000_000 * model_price[1]
        avg_cache_cost  = avg_cache_tokens  / 1_000_000 * model_price[2]

    print("\n===== Token Statistics =====")
    # print(f"Total input tokens:         {all_token_stat[0]:,}")
    # print(f"Total output tokens:        {all_token_stat[1]:,}")
    # print(f"Total cache tokens:         {all_token_stat[2]:,}")
    # print(f"Avg input tokens / sample:  {avg_input_tokens:.2f}")
    # print(f"Avg output tokens / sample: {avg_output_tokens:.2f}")
    # print(f"Total summary input tokens: {summary_token_stat[0]:,}")
    # print(f"Total summary output tokens:{summary_token_stat[1]:,}")
    # print(f"Total browse input tokens:  {browse_token_stat[0]:,}")
    # print(f"Total browse output tokens: {browse_token_stat[1]:,}")
    # print(f"Avg summary in / sample:    {summary_token_stat[0] / len(alldata) if alldata else 0:.2f}")
    # print(f"Avg summary out / sample:   {summary_token_stat[1] / len(alldata) if alldata else 0:.2f}")
    # print(f"Avg verify in / sample:     {verify_token_stat[0] / len(alldata) if alldata else 0:.2f}")
    # print(f"Avg verify out / sample:    {verify_token_stat[1] / len(alldata) if alldata else 0:.2f}")
    # print(f"Avg browse in / sample:     {browse_token_stat[0] / len(alldata) if alldata else 0:.2f}")
    # print(f"Avg browse out / sample:    {browse_token_stat[1] / len(alldata) if alldata else 0:.2f}")
    print(f"Avg input cost:             ${avg_input_cost:.4f}")
    print(f"Avg output cost:            ${avg_output_cost:.4f}")
    print(f"Avg cache cost:             ${avg_cache_cost:.4f}")
    print(f"Avg total cost:             ${avg_input_cost + avg_output_cost + avg_cache_cost:.4f}")
    print(f"Continue generation count:  {cnt_continue:,}, accuracy: {acc_continue / cnt_continue:.2f}" if cnt_continue > 0 else "No continued generation found")

    # query parallel stat
    print("\n===== Tool Parallel Analysis =====")

    if not query_parallel_dict:
        print("No query parallel data found.")
    else:
        avg_query_parallel = sum(k * v for k, v in query_parallel_dict.items()) / max(1, sum(query_parallel_dict.values()))
        print(f"Avg query parallel: {avg_query_parallel:.2f}")
        # for query_cnt, num_blocks in sorted(query_parallel_dict.items()):
        #     print(f"Query count: {query_cnt}, Number of blocks: {num_blocks}")

    if not url_parallel_dict:
        print("No URL parallel data found.")
    else:
        avg_url_parallel = sum(k * v for k, v in url_parallel_dict.items()) / max(1, sum(url_parallel_dict.values()))
        print(f"Avg URL parallel: {avg_url_parallel:.2f}")
        # for url_cnt, num_blocks in sorted(url_parallel_dict.items()):
        #     print(f"URL count: {url_cnt}, Number of blocks: {num_blocks}")

    # v9 adaptive memory stat
    print("\n===== Adaptive Memory Analysis =====")
    use_cnt = len(use_ad_sum_score)
    print(f"Adaptive summary used in {use_cnt} samples, {use_cnt / len(alldata):.2%} of total samples.")
    avg_ad_sum_cnt = np.mean(ad_memory_cnt_list) if ad_memory_cnt_list else 0
    # print(f"Avg adaptive summary count: {avg_ad_sum_cnt:.2f}")
    print(f"Use adaptive summary score: {np.mean(use_ad_sum_score) if use_ad_sum_score else 0.0:.4f}")
    print(f"No use adaptive summary score: {np.mean(no_use_ad_sum_score) if no_use_ad_sum_score else 0.0:.4f}")

    # overfitting analysis
    print("\n===== Overfitting Analysis =====")
    print(f"Tool code found in {tool_code_cnt} samples, {tool_code_cnt / len(alldata):.2%} of total samples.")
    # verify dict
    verify_dict_output = {k:[sum(v), len(v)] for k, v in verify_decision_score_dict.items()}

    ret_results = {
        "acc": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "total": len(alldata),
        "ans/none": none_ans_cnt,
        "ans/non_none_ans_acc": sum(non_none_ans_score_list) / len(non_none_ans_score_list) if non_none_ans_score_list else -1,
        "query/avg": avg_query_cnt,
        "query/total": sum(all_query_cnt_list),
        "query/budget": budget_search,
        "query/use_all_budget": cnt_use_all_search_bgts / len(all_query_cnt_list) if len(all_query_cnt_list) > 0 else -1,
        "query/sample_no_result_ratio": search_no_result_cnt / len(alldata) if alldata else -1,
        "query/query_no_result_ratio": query_no_result_cnt / sum(all_query_cnt_list) if sum(all_query_cnt_list) > 0 else -1,
        "query/sample_api_error_ratio": sample_api_error_cnt / len(alldata) if alldata else -1,
        "query/query_api_error_ratio": query_api_error_cnt / sum(all_query_cnt_list) if sum(all_query_cnt_list) > 0 else -1,
        "query/correct_ans_query_cnt_avg": mean(correct_ans_query_num_list) if len(correct_ans_query_num_list) > 0 else 0.0,
        "query/wrong_ans_query_cnt_avg": mean(wrong_ans_query_num_list) if len(wrong_ans_query_num_list) > 0 else 0.0,
        "query/parallel_avg": avg_query_parallel if query_parallel_dict else -1,
        "url/avg": avg_url_cnt,
        "url/total": sum(all_url_cnt_list),
        "url/budget": budget_browse,
        "url/use_all_budget": cnt_use_all_browse_bgts / len(all_url_cnt_list) if len(all_url_cnt_list) > 0 else -1,
        "url/sample_no_result_ratio": browse_no_result_cnt / len(alldata) if alldata else -1,
        "url/url_no_result_ratio": url_no_result_cnt / sum(all_url_cnt_list) if sum(all_url_cnt_list) > 0 else -1,
        "url/correct_ans_url_cnt_avg": mean(correct_ans_url_num_list) if len(correct_ans_url_num_list) > 0 else 0.0,
        "url/wrong_ans_url_cnt_avg": mean(wrong_ans_url_num_list) if len(wrong_ans_url_num_list) > 0 else 0.0,
        "url/parallel_avg": avg_url_parallel if url_parallel_dict else -1,
        "token/input_total": all_token_stat[0],
        "token/output_total": all_token_stat[1],
        "token/cache_total": all_token_stat[2],
        "token/input_avg": avg_input_tokens,
        "token/output_avg": avg_output_tokens,
        "token/cache_avg": avg_cache_tokens,
        "token/summary_input_total": summary_token_stat[0],
        "token/summary_output_total": summary_token_stat[1],
        "token/summary_cache_total": summary_token_stat[2],
        "token/summary_input_avg": (summary_token_stat[0] / len(alldata)) if alldata else 0.0,
        "token/summary_output_avg": (summary_token_stat[1] / len(alldata)) if alldata else 0.0,
        "token/summary_cache_avg": (summary_token_stat[2] / len(alldata)) if alldata else 0.0,
        "token/verify_input_total": verify_token_stat[0],
        "token/verify_output_total": verify_token_stat[1],
        "token/verify_cache_total": verify_token_stat[2],
        "token/verify_input_avg": (verify_token_stat[0] / len(alldata)) if alldata else 0.0,
        "token/verify_output_avg": (verify_token_stat[1] / len(alldata)) if alldata else 0.0,
        "token/verify_cache_avg": (verify_token_stat[2] / len(alldata)) if alldata else 0.0,
        "token/browse_input_total": browse_token_stat[0],
        "token/browse_output_total": browse_token_stat[1],
        "token/browse_cache_total": browse_token_stat[2],
        "token/browse_input_avg": (browse_token_stat[0] / len(alldata)) if alldata else 0.0,
        "token/browse_output_avg": (browse_token_stat[1] / len(alldata)) if alldata else 0.0,
        "token/browse_cache_avg": (browse_token_stat[2] / len(alldata)) if alldata else 0.0,
        "price/input": avg_input_cost,
        "price/output": avg_output_cost,
        "price/cache": avg_cache_cost,
        "price/total": avg_input_cost + avg_output_cost + avg_cache_cost,
        "continue/count": cnt_continue,
        "continue/accuracy": acc_continue / cnt_continue if cnt_continue > 0 else -1,
        "verify/count": cnt_verify,
        "verify/success": cnt_verify_success,
        "verify/accuracy": acc_verify / cnt_verify if cnt_verify > 0 else -1,
        "verify/accuracy_in_success": acc_verify_success / cnt_verify_success if cnt_verify_success > 0 else -1,
        "verify/decision": verify_dict_output,
        "data_path": data_path,
        "adsum/avg_count": avg_ad_sum_cnt,
        "adsum/use_score": np.mean(use_ad_sum_score) if use_ad_sum_score else 0.0,
        "adsum/no_use_score": np.mean(no_use_ad_sum_score) if no_use_ad_sum_score else 0.0,
    }

    if plot:
        # run_name for figure title
        run_name = data_path
        for pivot in ["flash/", "gemini-2.5-flash/", "outputs/"]:
            if pivot in run_name:
                run_name = run_name.split(pivot, 1)[-1]
        run_name = run_name.rsplit(".jsonl", 1)[0]

        search_plot_path = plot_query_bins_with_accuracy(
            correct_ans_query_num_list,
            wrong_ans_query_num_list,
            bin_size=5,
            max_x=max(all_query_cnt_list) if all_query_cnt_list else 50,
            save_dir="figs/analysis",
            filename=f"search_{run_name}.png"
        )
        print(f"search-plot saved to: {search_plot_path}")

        browse_plot_path = plot_query_bins_with_accuracy(
            correct_ans_url_num_list,
            wrong_ans_url_num_list,
            bin_size=5,
            max_x=max(all_url_cnt_list) if all_url_cnt_list else 50,
            save_dir="figs/analysis",
            filename=f"browse_{run_name}.png"
        )
        print(f"browse-plot saved to: {browse_plot_path}")
    return ret_results

# --------- CLI ---------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze search results in a JSONL file.")
    parser.add_argument("-t","--total", type=int, default=-1, help="Number of samples to analyze. Use -1 for all.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the JSONL file.")
    parser.add_argument("-w", "--wandb", action="store_true", help="Log results to wandb.")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot query distribution.")
    args = parser.parse_args()

    data_path = args.data_path
    # extract budgets safely
    budget_search = extract_budget_from_path(data_path, "bgts")
    budget_browse = extract_budget_from_path(data_path, "bgtb")

    results = analyze_tool_results(data_path, args.total, budget_search, budget_browse, args.plot)

    if args.wandb:
        import wandb
        run_name = data_path
        if "gemini-2.5-flash" in data_path:
            for pivot in ["flash/", "gemini-2.5-flash/", "outputs/"]:
                if pivot in run_name:
                    run_name = run_name.split(pivot, 1)[-1]
            run_name = "flash_" + run_name
        elif "gemini-2.5-pro" in data_path:
            for pivot in ["pro/", "gemini-2.5-pro/", "outputs/"]:
                if pivot in run_name:
                    run_name = run_name.split(pivot, 1)[-1]
            run_name = "pro_" + run_name
        run_name = run_name.rsplit(".jsonl", 1)[0]
        wandb.init(project="browsecomp", name=f"{run_name}_post_analysis")
        wandb.log(results)
        wandb.finish()
