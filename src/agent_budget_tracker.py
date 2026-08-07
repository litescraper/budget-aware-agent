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
import re
import json
import time
import ast
import asyncio
import aiohttp
import argparse
import concurrent.futures
from typing import Any, List, Dict, Optional, Union
import itertools
import uuid
import multiprocessing as mp
import wandb

from prompts_budget_tracker import *
from helpers import *
from llm import llm_text_gen_conversation, check_credentials

# ✅ Changed to import SQLite version of search tool
# If get_query is already defined elsewhere, you can only import search_with_sqlite_cache_async
from tool_google_search import search_with_sqlite_cache_async, get_query, check_search_credentials
from tool_browse import BrowseTool

# Assume load_dotenv() is called and env variables are set
from dotenv import load_dotenv
load_dotenv()

TOOL_CALL_TAG = os.environ.get("TOOL_CALL_TAG", "tool_code")  # Default to 'tool_call' if not set
ITER_LIMIT = 100

# Stop sequences used by this agent
STOP_SEQUENCES = [f"</{TOOL_CALL_TAG}>", f"\n</{TOOL_CALL_TAG}>", "</tool_response>", "</answer>", "\n</answer>"]
# ==============================================================================
#  SECTION 1: ASYNC WORKER CODE 
#  This is the logic for a SINGLE process/question.
# ==============================================================================

BROWSE_CLIENT = None  # per-process singleton





# ===== Core Logic for a Single Data Entry =====
async def run_inference_on_question_async(
    session: aiohttp.ClientSession, data_entry: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """
    It manages the iterative workflow for a single data entry using conversation format.
    """
    verbose = False
    idx = data_entry['uid']
    problem = data_entry['problem']
    print(f"[PID:{os.getpid()}] Starting (Q#{idx}): {problem[:30]}...")

    SEARCH_BUDGET = args.search_budget
    BROWSE_BUDGET = args.browse_budget
    all_queries, all_urls = [], []

    # Budget template and initial budget info
    budget_template = (
        "\n<budget>\n"
        "Query Budget Used: {q_used}, Query Budget Remaining: {q_rem}, "
        "URL Budget Used: {u_used}, URL Budget Remaining: {u_rem}\n"
        "Make the best use of the available resources.\n"
        "</budget>\n"
    )
    initial_budget = budget_template.format(
        q_used=0, q_rem=SEARCH_BUDGET, u_used=0, u_rem=BROWSE_BUDGET
    )
    
    # Initialize conversation with the question and budget info
    conversation = [
        {"role": "user", "content": f"Question:\n{problem.strip()}{initial_budget}"}
    ]
    
    cnt = 0
    generated_text, final_answer, tmp_answer, full_trajectory = "", None, None, f"Question:\n{problem.strip()}\n"

    cache_hit_est = CacheHitEstimator(args.model)
    all_token_counts = [0, 0, 0] # input, output, cache_hit
    browse_token_counts = [0, 0, 0]

    last_ans_round = -1
    all_answers = []

    # gemini no answer 
    cnt_fail = 0
    # in response, no query generated
    cnt_no_tool = 0
    # Use the standard system instruction since we're now using proper Content objects with roles
    cur_si = pr_system_instruction_initial().strip()

    flag_has_answer = False
    none_ans_cnt = 0
    none_ans_threshold = args.none_ans_threshold
    # current response state
    cur_state = "reason"  # reason, answer, summary
    flag_continue = False # whether this question has continue generation step

    while cnt < ITER_LIMIT and none_ans_cnt < none_ans_threshold and cnt_no_tool < none_ans_threshold and (len(all_queries) < SEARCH_BUDGET and len(all_urls) < BROWSE_BUDGET):

        # if args.flag_cont_gen:
        #     if len(all_queries) >= SEARCH_BUDGET and len(all_urls) >= BROWSE_BUDGET:
        #         print(f"[PID:{os.getpid()}] Both search and browse budgets exhausted. Breaking.", flush=True)
        #         break

        cnt += 1

        print(f"[PID:{os.getpid()}] 💨 ====== Q#{idx} Round {cnt} | State: {cur_state} | Queries: {len(all_queries)}/{SEARCH_BUDGET} | URLs: {len(all_urls)}/{BROWSE_BUDGET} | NoneAnsCnt: {none_ans_cnt} | NoToolCnt: {cnt_no_tool} ======", flush=True)

        full_trajectory += f"\n\n[[Round {cnt}]]\n"
        full_trajectory += f"\n[[SI: {cur_si[:200]}]]\n"
        
        # Generate response using conversation format
        response, token_counts = llm_text_gen_conversation(
            conversation=conversation,
            model=args.model,
            args=args,
            si=cur_si,
            stop_sequences=STOP_SEQUENCES,
        )
        
        if isinstance(token_counts, list) and len(token_counts) == 2:
            cache_hit_cnt = cache_hit_est.estimate_and_update(str(conversation))
            all_token_counts[0] += (token_counts[0] - cache_hit_cnt)
            all_token_counts[1] += token_counts[1]
            all_token_counts[2] += cache_hit_cnt
        
        # this means gemini exceeds the max output tokens or has error, we will directly generate answer
        if not response or not response.text:
            cnt_fail += 1
            if cnt_fail >= 2:
                print(f"[PID:{os.getpid()}]♊️ [Gemini] Failed to get a response from the model. Aborted.", flush=True)
                cur_state = "abort"
                break
            print(f"[PID:{os.getpid()}]♊️ [Gemini] Failed to get a response from the model. Directly generate answer...", flush=True)

            print(f"=====================gemini conversation")
            print(f"SI: {cur_si.strip()}")
            # print(f"Conversation: {conversation}")

            cur_state = "answer"
            # Add final answer prompt to conversation
            conversation.append({"role": "user", "content": pr_prompt_generate_final_answer().strip()})
            cur_si = pr_system_instruction_generate_final_answer().strip()
            full_trajectory += "\n[[Failed to get response, directly generate answer]]\n"
            continue
        
        generated_text = response.text.strip()

        
        

        # end_tag could be <answer> or <tool_call>
        end_tag = find_later_tag(generated_text)
        if end_tag:
            generated_text += f"\n{end_tag.replace('<', '</')}\n"
        # === Add model response to conversation ===
        conversation.append({"role": "model", "content": generated_text})

        # end_tag is <answer>
        if end_tag == "<answer>" and "<answer>" in generated_text:
            matches = re.findall(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
            if matches:
                tmp_answer = matches[-1].strip()
                all_answers.append([cnt, tmp_answer])
                last_ans_round = cnt
                final_answer = tmp_answer

                full_trajectory += generated_text
                # if budget not exhausted, keep exploring
                
                # sequential scaling: always continue generation
                if args.flag_cont_gen and (len(all_queries) < SEARCH_BUDGET or len(all_urls) < BROWSE_BUDGET) and none_ans_cnt < none_ans_threshold:
                    print(f"[PID:{os.getpid()}] ▶️ Continue working...", flush=True)
                    full_trajectory += f"\n[[answer tag found, but budget not exhausted, continue working]]\n"
                    # generated_text = generated_text.replace("<answer>", "").replace("</answer>", "")
                    
                    # Add continuation prompt to conversation
                    remaining_tools = ""
                    if len(all_queries) < SEARCH_BUDGET or len(all_urls) < BROWSE_BUDGET:
                        remaining_tools = "Wait, you still have remaining tool budget — use more search and browse tools to explore different information sources before concluding."

                    # Add budget info to continuation prompt
                    budget_info = budget_template.format(
                        q_used=len(all_queries),
                        q_rem=SEARCH_BUDGET-len(all_queries),
                        u_used=len(all_urls),
                        u_rem=BROWSE_BUDGET-len(all_urls)
                    )
                    remaining_tools += budget_info
                    
                    conversation.append({"role": "user", "content": remaining_tools})
                    cur_si = pr_system_instruction_initial().strip()
                    cur_state = "reason"
                    none_ans_cnt += 1
                    flag_continue = True
                    full_trajectory += f"\n[[Remaining tools: {remaining_tools.strip()}]]\n"
                    continue
                else:
                    if len(all_queries) >= SEARCH_BUDGET:
                        print(f"[PID:{os.getpid()}] len(all_queries) >= budget", flush=True)
                    if len(all_urls) >= BROWSE_BUDGET:
                        print(f"[PID:{os.getpid()}] len(all_urls) >= budget", flush=True)
                    if none_ans_cnt >= none_ans_threshold:
                        print(f"[PID:{os.getpid()}] none_ans_cnt >= {none_ans_threshold}", flush=True)
                    print(f"[PID:{os.getpid()}] none ans: {none_ans_cnt}", flush=True)
                    print(f"[PID:{os.getpid()}] 🎉 Final answer found: {final_answer}\nGround truth: {data_entry['answer']}", flush=True)
                    break

        # extract tool from the generated text
        try:
            tool_call = extract_tool_call(generated_text)
        except Exception as e:
            print(f"[PID:{os.getpid()}] ❗ Error extracting tool call: {e}\n{generated_text}", flush=True)
            tool_call = {}

        tool_name = tool_call.get('name', '')
        tool_args = tool_call.get('arguments', {})
        print(f"🔍 Tool call detected: {tool_name} with args {tool_args}")

        # =======================================
        # =========Tool execution logic==========
        # =======================================
        tool_response_str = ""
        if tool_name == "search":
            if "query" in tool_args:
                query_list = tool_args.get('query', [])
            elif "query" in str(tool_call):
                query_list = tool_call.get('query', [])
            else:
                query_list = []
            query_list = [q.strip() for q in query_list]
            query_num_before_truncate = len(query_list)
            if len(query_list) > SEARCH_BUDGET - len(all_queries):
                query_list = query_list[:SEARCH_BUDGET - len(all_queries)]
                tool_args['query'] = query_list
                print(f"🔍 Truncated queries to fit within search budget: {query_list}")

            print(f"[PID:{os.getpid()}] ⭕️ Extracted query: {query_list}", flush=True)
            
            if query_list:
                cnt_no_tool = 0
                # === query optimization module ===
                if args.flag_optimize_query:
                    for q_i, query in enumerate(query_list):
                        if not is_good_query(query):
                            old_query = query
                            query = query.replace('"', '').replace("'", "").strip()
                            query_list[q_i] = query
                            print(f"[PID:{os.getpid()}] 🔍 Optimized query from '{old_query}' to '{query}'", flush=True)
                # perform search
                all_queries.extend(query_list)
                search_tasks = [
                    search_with_sqlite_cache_async(
                        session,
                        q,
                        num_results=args.n_query_results,
                    )
                    for q in query_list
                ]
                search_results_str_list = await asyncio.gather(*search_tasks)
                for search_i in range(len(query_list)):
                    tool_response_str += f"Query: {query_list[search_i]}\nSearch results:\n{search_results_str_list[search_i]}\n"
            else:
                if query_num_before_truncate > 0:
                    print(f"[PID:{os.getpid()}] ❗ All queries truncated due to budget limit. {generated_text}", flush=True)
                    cnt_no_tool = 0
                    tool_response_str = "No query budget left."
                else:
                    print(f"[PID:{os.getpid()}] ❗ No query found in the tool search. {generated_text}", flush=True)
                    cnt_no_tool += 1
        
        elif tool_name == "browse":
            if "url" in tool_args:
                url_list = tool_args.get('url', [])
            elif "url" in str(tool_call):
                url_list = tool_call.get('url', [])
            else:
                url_list = []

            if "goal" in tool_args:
                browse_goal = tool_args.get('goal', '')
            elif "goal" in str(tool_call):
                browse_goal = tool_call.get('goal', '')
            else:
                browse_goal = ''

            if type(url_list) == str:
                url_list = [url_list]
            
            url_list = [u for u in url_list if len(u) > 5 and "." in u]
            url_num_before_truncate = len(url_list)
            if len(url_list) > BROWSE_BUDGET - len(all_urls):
                url_list = url_list[:BROWSE_BUDGET - len(all_urls)]
                tool_args['url'] = url_list
                print(f"🔍 Truncated urls to fit within browse budget: {url_list}")

            url_list = [u.strip() for u in url_list if u.strip()]
            # todo: add flash as summarizer
            if url_list:
                cnt_no_tool = 0
                all_urls.extend(url_list)

                global BROWSE_CLIENT
                try:
                    page_map = await BROWSE_CLIENT.browse(url_list, session)
                except Exception as e:
                    page_map = {u: {"content": f"[Error] {e}", "source": "error"} for u in url_list}
                
                # 1) Separate summarizable pages from error pages
                summarize_pages = []   # [url]
                error_pages = []       # [url]

                webpage_summary_prompts = []
                for u in url_list:
                    r = page_map.get(u, {"content": "", "source": "error"})
                    src = r.get("source", "error").strip()
                    content = r.get("content", "").strip()
                    # build gemini prompt, and use async to get gemini summary
                    # input should include both url and goal
                    if src and src != "error" and content:
                        summarize_pages.append(u)
                        one_p = pr_browse_summary().format(webpage_content=truncate(content), goal=browse_goal)
                        webpage_summary_prompts.append(one_p)
                    else:
                        error_pages.append(u)

                # 2) summarize by gemini
                if webpage_summary_prompts:
                    print(f"[PID:{os.getpid()}] 📝 Summarizing {len(webpage_summary_prompts)} pages with Gemini...", flush=True)

                    # Concurrently call the synchronous LLM call in a thread pool to avoid blocking the event loop
                    async def _summ_one(u: str, prompt_sum: str):
                        # Execute synchronous function using to_thread
                        resp, tks = await asyncio.to_thread(
                            llm_text_gen_conversation,
                            [{"role": "user", "content": prompt_sum}],          # conversation
                            args.browse_model if args.flag_browse_flash else args.model,  # model
                            args,                # args
                            pr_system_instruction_browse_summary(),  # si
                            STOP_SEQUENCES,
                        )
                        # Count tokens (you can remove these lines if not needed)
                        # nonlocal all_token_counts
                        if isinstance(tks, list) and len(tks) == 2:
                            cache_hit_cnt = cache_hit_est.estimate_and_update(pr_system_instruction_browse_summary() + "\n\n\n\n\n" + prompt_sum)
                            all_token_counts[0] += (tks[0] - cache_hit_cnt)
                            all_token_counts[1] += tks[1]
                            all_token_counts[2] += cache_hit_cnt
                            browse_token_counts[0] += (tks[0] - cache_hit_cnt)
                            browse_token_counts[1] += tks[1]
                            browse_token_counts[2] += cache_hit_cnt

                        text = (resp.text or "").strip() if resp else ""
                        if not text:
                            text = f"Browse {u}:\nIrrelevant or empty."
                        elif not text.lower().startswith("browse"):
                            # Normalize output format for easier concatenation
                            text = f"Browse {u}:\n{text}"
                        return u, text

                    # Run all summaries concurrently
                    try:
                        url_and_text_list = await asyncio.gather(
                            *[_summ_one(u, p) for u, p in zip(summarize_pages, webpage_summary_prompts)],
                            return_exceptions=True
                        )
                    except Exception as e:
                        url_and_text_list = [e]

                    summaries_per_url = {}
                    for item in url_and_text_list:
                        if isinstance(item, Exception):
                            # Should not reach here (because return_exceptions=True), but just in case
                            continue
                        u, summ = item
                        summaries_per_url[u] = summ

                    # Write only the summary body back to <tool_response>
                    tool_response_str = f"Goal for browsing: {browse_goal}\n---\n"
                    # Keep the same order as original url_list; provide placeholder for error pages
                    for u in url_list:
                        if u in summaries_per_url:
                            tool_response_str += summaries_per_url[u].rstrip() + "\n---\n"
                        elif u in error_pages:
                            tool_response_str += f"Browse {u}:\nNo content.\n---\n"
                    # page_results_str += f"URL: {u}\nSource: {src}\nContent:\n{_truncate(str(content), 15000)}\n---\n"

            else:
                if url_num_before_truncate > 0:
                    print(f"[PID:{os.getpid()}] ❗ All urls truncated due to budget limit. {generated_text}", flush=True)
                    cnt_no_tool = 0
                    tool_response_str = "No url budget left."
                else:
                    print(f"[PID:{os.getpid()}] ❗ No url found in the tool browse. {generated_text}", flush=True)
                    cnt_no_tool += 1

        else:
            print(f"[PID:{os.getpid()}] ❗ No valid tool found in the response. {generated_text}", flush=True)
            cnt_no_tool += 1

        # === remove previous tool responses ===
        if args.flag_remove_tool_responses:
            # Remove tool responses, but still keep the round
            new_conversation = []
            for msg in conversation:
                if msg['role'] == 'user' and '<tool_response>' in msg['content']:
                    new_conversation.append({"role": "user", "content": f"<tool_response>\nOmitted\n</tool_response>"})
                else:
                    new_conversation.append(msg)
            conversation = new_conversation
        
        # Add budget info to conversation
        budget_info = budget_template.format(
            q_used=len(all_queries),
            q_rem=SEARCH_BUDGET-len(all_queries),
            u_used=len(all_urls),
            u_rem=BROWSE_BUDGET-len(all_urls)
        )

        # Add tool response to conversation as user message
        if tool_response_str:
            conversation.append({"role": "user", "content": f"<tool_response>\n{tool_response_str}\n</tool_response>\n{budget_info}"})
            full_trajectory += f"\n\n{generated_text}<tool_response>\n{tool_response_str}\n</tool_response>\n\n"
        else:
            print(f"[PID:{os.getpid()}] ❗ No tool response.", flush=True)
            conversation.append({"role": "user", "content": f"<tool_response>\nNo tool response.\n</tool_response>\n{budget_info}"})
            full_trajectory += f"\n\n{generated_text}\n\n"

        

        
        # === conversation management ===
        # Keep conversation as is for now, since context length is not the issue

        # === if no tool, consider generate answer ===
        if tool_response_str == "" and ("answer:" in generated_text.lower() or "answer is" in generated_text.lower()):
            full_trajectory += "\n[[tool is None, and answer found]]\n"
            cur_state = "answer"
            print(f"[PID:{os.getpid()}] ℹ️ No tool response, but answer found, switch to answer state.", flush=True)

        if cur_state == "answer":
            conversation.append({"role": "user", "content": pr_prompt_generate_final_answer().strip()})
            cur_si = pr_system_instruction_generate_final_answer().strip()

        if verbose:
            print(f"[PID:{os.getpid()}] ------ Round {cnt} conversation ------\n{conversation}", flush=True)

        # for scaling
        if args.flag_cont_gen:
            remaining_tools = ""
            if cnt_no_tool >= 1 and len(all_queries) < SEARCH_BUDGET and len(all_urls) < BROWSE_BUDGET:
                remaining_tools = "Wait, you still have remaining tool budget — use more search and browse tools to explore different information sources before concluding."
                
            if remaining_tools:
                # Add budget info to remaining tools prompt
                budget_info = budget_template.format(
                    q_used=len(all_queries),
                    q_rem=SEARCH_BUDGET-len(all_queries),
                    u_used=len(all_urls),
                    u_rem=BROWSE_BUDGET-len(all_urls)
                )
                remaining_tools += budget_info
                conversation.append({"role": "user", "content": remaining_tools})
                full_trajectory += f"\n[[Remaining tools: {remaining_tools.strip()}]]\n"
    
    # If we still don't have a final answer after all rounds, force generate it
    if last_ans_round != cnt:
        print(f"[PID:{os.getpid()}] No final answer found after {len(all_queries)}(bgt: {SEARCH_BUDGET}) searches and {len(all_urls)} (bgt: {BROWSE_BUDGET}) browses. Using the last generated text as final answer.", flush=True)
        if cur_state == "abort":
            # sequence is too long, use the last generated_text as input
            conversation = [{"role": "user", "content": f"Question: {problem.strip()}\n{generated_text.strip()}"}]

        if not any(pr_prompt_generate_final_answer().strip() in msg['content'] for msg in conversation if msg['role'] == 'user'):
            conversation.append({"role": "user", "content": pr_prompt_generate_final_answer().strip()})
            full_trajectory += "\n[[final answer is none and budget exhausted]]\n"
            full_trajectory += f"\n{pr_prompt_generate_final_answer().strip()}"
        
        # Generate the final answer
        cur_si = pr_system_instruction_generate_final_answer().strip()
        full_trajectory += f"\n{cur_si[:200]}\n"
        response, token_counts = llm_text_gen_conversation(
            conversation=conversation,
            model=args.model,
            args=args,
            si=cur_si,
            stop_sequences=STOP_SEQUENCES,
        )
        if isinstance(token_counts, list) and len(token_counts) == 2:
            cache_hit_cnt = cache_hit_est.estimate_and_update(str(conversation))
            all_token_counts[0] += (token_counts[0] - cache_hit_cnt)
            all_token_counts[1] += token_counts[1]
            all_token_counts[2] += cache_hit_cnt
        if response and response.text:
            generated_text = response.text.strip()
            
            if "<answer>" in generated_text:
                generated_text += "\n</answer>\n"
                match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
                if match:
                    final_answer = match.group(1).strip()
                    print(f"[PID:{os.getpid()}] 🎉 Final answer found: {final_answer}\nGround truth: {data_entry['answer']}", flush=True)
            
            if final_answer is None:
                final_answer = generated_text
            full_trajectory += "\n[[final answer extracted from the final round]]\n"
            full_trajectory += generated_text

            all_answers.append([cnt, final_answer])
        else:
            full_trajectory += "\n[[Tool budget exhausted. No final answer generated.]]\n"

    if final_answer is None:
        final_answer = "None"

    print(f"[PID:{os.getpid()}] Finished Q#{idx}", flush=True)

    if len(all_queries) >= SEARCH_BUDGET and len(all_urls) >= BROWSE_BUDGET:
        stop_reason = "budget"
    elif none_ans_cnt >= none_ans_threshold:
        stop_reason = "none_ans"
    elif cnt >= ITER_LIMIT:
        stop_reason = "iter_limit"
    else:
        stop_reason = "other"
    return {
        "index": idx,
        "question": problem,
        "gold_answer": data_entry.get("answer", ""), # Include if available
        "predicted_answer": final_answer,
        "query_count": len(all_queries),
        "queries": all_queries,
        "url_count": len(all_urls),
        "urls": all_urls,
        "full_trajectory": full_trajectory,
        "conversation": conversation,  # Add conversation history to results
        "all_token_counts": all_token_counts,
        "browse_token_counts": browse_token_counts,
        "flag_continue": flag_continue,
        "none_ans_cnt": none_ans_cnt,
        "stop_reason": stop_reason,
        "all_answers": all_answers
    }

# ==============================================================================
#  SECTION 2: PROCESS MANAGEMENT & EXECUTION (Updated Logic)
#  This section now distributes chunks of data entries to each process.
# ==============================================================================

def process_chunk_entry_point(chunk_and_args: tuple) -> str:
    """
    Synchronous wrapper for a process. It saves results to a temporary file.
    Returns the path to the temporary file.
    """
    # ✅ Removed shared_cache / lock
    data_entries, args, temp_dir = chunk_and_args


    # Init per-process browse tool (shared SQLite via args.db_path)
    global BROWSE_CLIENT
    BROWSE_CLIENT = BrowseTool(cache_db=args.browse_db_path)
    print(f"[PID:{os.getpid()}] Initialized BrowseTool with DB", flush=True)

    # Create a unique temporary file for this process's results
    temp_file_path = os.path.join(temp_dir, f"temp_results_{os.getpid()}_{uuid.uuid4()}.jsonl")
    print(f"[PID:{os.getpid()}] Assigned chunk of {len(data_entries)} entries. Writing to {temp_file_path}", flush=True)

    # Run the async workflow for the chunk
    asyncio.run(run_workflow_for_chunk(data_entries, args, temp_file_path))
    return temp_file_path

async def run_workflow_for_chunk(
    data_entries: List[Dict[str, Any]], args: argparse.Namespace, temp_file_path: str
):
    """
    Creates tasks for each data entry and writes results to the temp file as they complete.
    """
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes total timeout

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            run_inference_on_question_async(session, entry, args)
            for entry in data_entries
        ]

        with open(temp_file_path, 'w') as f:
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    f.write(json.dumps(result) + '\n')
                    f.flush()  # Ensure it's written immediately
                except Exception as e:
                    print(f"[PID:{os.getpid()}] ❌ Error in task: {e}", flush=True)

    print(f"[PID:{os.getpid()}] 🚀 Finished chunk. All results saved to {temp_file_path}.", flush=True)

def parser():
    parser = argparse.ArgumentParser(description="Run async inference with conversation format.")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash",
                    help="LiteLLM model name, e.g. gemini/gemini-2.5-pro, anthropic/claude-sonnet-4-5, openai/gpt-5.")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--search_budget", type=int, default=10, help="Maximum number of search iterations per question.")
    parser.add_argument("--browse_budget", type=int, default=10, help="Maximum number of browse iterations per question.")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--tag", type=str, default="v1", help="Tag for the output file.")
    parser.add_argument("--max_output_tokens", type=int, default=65535)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # search config
    parser.add_argument("--flag_gemini_no_think", action='store_true', help="Whether to remove the 'think' part from Gemini responses.")
    parser.add_argument("--flag_browse_flash", action='store_true', help="use the cheaper --browse_model for page summaries.")
    parser.add_argument("--browse_model", type=str, default="gemini/gemini-2.5-flash",
                        help="Model used for page summaries when --flag_browse_flash is set.")
    parser.add_argument("--flag_remove_tool_responses", action='store_true', help="Whether to remove search results from the trajectory.")
    parser.add_argument("--flag_optimize_query", action='store_true', help="Whether to optimize the search query before searching.")
    parser.add_argument("--flag_cont_gen", action='store_true', help="Whether to continue generation after the none answer.")
    parser.add_argument("--n_query_results", type=int, default=5)
    parser.add_argument("--none_ans_threshold", type=int, default=3, help="Threshold for none answer before stopping the search.")
    parser.add_argument("--search_backend", type=str, default=os.environ.get("SEARCH_BACKEND", "cse"), choices=["cse", "serper"],
                        help="Web search backend: 'cse' (Google Custom Search, paper setting) or 'serper' (serper.dev).")

    # parallel params
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent tasks to run.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to evaluate the answers after inference.")
    parser.add_argument("--wandb", action='store_true', help="Log evaluation results to Weights & Biases (requires `wandb login`).")
    parser.add_argument("--wandb_project", type=str, default="browsecomp", help="WandB project name.")

    # ✅ New: SQLite cache parameters (replacing JSON cache)
    parser.add_argument("--browse_db_path", type=str, default="outputs/cache/browse_cache_0909.sqlite3",
                        help="Path to the shared SQLite cache file.")
    parser.add_argument("--search_ttl", type=int, default=0,
                        help="Cache TTL in seconds (0 means no expiration).")

    # Compatible with old parameters (no longer used)
    parser.add_argument("--cache_path", type=str, default="", help="(Deprecated) JSON cache path, unused with SQLite.")
    return parser

def main():
    """
    Main function that uses a process pool to run your workflows in true parallel.
    """

    args = parser().parse_args()
    check_credentials(args.model)
    os.environ["SEARCH_BACKEND"] = args.search_backend
    check_search_credentials()
    print(f"In main(), getting data...")

    # prepare data
    all_data = []
    if not args.data_path:
        raise SystemExit("--data_path is required. Build the dataset files first with "
                         "`python scripts/prepare_data.py` (see the Datasets section of README.md).")
    with open(args.data_path, 'r') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(all_data)} records from {args.data_path}")

    if args.end == -1:
        args.end = len(all_data)

    print(f"Using data subset from {args.begin} to {args.end} (exclusive).", flush=True)
    all_data = all_data[args.begin:args.end]
    
    output_dir = f"outputs/agent_budget_tracker/{args.model.replace('/', '_')}/{args.tag}"
    os.makedirs(output_dir, exist_ok=True)
    # Save path for the final results
    save_path = os.path.join(output_dir, 
        f"t{args.temperature}_bgts{args.search_budget}_bgtb{args.browse_budget}_nqr{args.n_query_results}_none{args.none_ans_threshold}_b{args.begin}_e{args.end}.jsonl"
    )

    if "flash" in args.model:
        model_alias = "flash"
    elif "pro" in args.model:
        model_alias = "pro"
    else:
        model_alias = args.model.split("/")[-1]
    wandb_name = f"{model_alias}_{args.tag}_t{args.temperature}_bgts{args.search_budget}_bgtb{args.browse_budget}_nqr{args.n_query_results}_none{args.none_ans_threshold}_b{args.begin}_e{args.end}_conv"

    if args.flag_remove_tool_responses:
        save_path = save_path.replace(".jsonl", "_rsr.jsonl")
        wandb_name += "_rsr"
    if args.flag_optimize_query:
        save_path = save_path.replace(".jsonl", "_oq.jsonl")
        wandb_name += "_oq"
    if args.flag_cont_gen:
        save_path = save_path.replace(".jsonl", "_cg.jsonl")
        wandb_name += "_cg"
    if args.flag_browse_flash:
        save_path = save_path.replace(".jsonl", "_bf.jsonl")
        wandb_name += "_bf"
    if args.flag_gemini_no_think:
        save_path = save_path.replace(".jsonl", "_nt.jsonl")
        wandb_name += "_nt"

    temp_dir = os.path.join(output_dir, f"temp_{args.tag}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)

    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Exiting to avoid overwriting.", flush=True)
        print(f"rm {save_path}")
        print(f"rm {save_path.replace('.jsonl', '_evalv1.jsonl')}")
        exit()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Results will be saved to {save_path}, time starts", flush=True)
    start_time = time.time()

    # ✅ Removed old JSON Cache Setup Block (no longer needed)

    print(f"🏃 Running multi-process on {len(all_data)} data with concurrency {args.concurrency} ...", flush=True)
    
    num_processes = args.concurrency
    # --- Distribute data into contiguous chunks for each process ---
    n_entries = len(all_data)
    # This calculation ensures chunks are as evenly sized as possible
    chunk_size = (n_entries + num_processes - 1) // num_processes
    data_chunks = [
        all_data[i : i + chunk_size]
        for i in range(0, n_entries, chunk_size)
    ]

    print(f"--- Starting {n_entries} workflows on up to {num_processes} processes ---", flush=True)
    
    temp_file_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Pass parameters: no longer includes shared_cache, lock
        chunk_and_args_iterator = zip(
            data_chunks,
            itertools.repeat(args),
            itertools.repeat(temp_dir),
        )
        temp_file_paths = list(executor.map(process_chunk_entry_point, chunk_and_args_iterator))

    # --- Merge, Sort, and Save Final Results ---
    print("\n\n--- ✅ All processes finished. Merging results... ---", flush=True)
    all_results = []
    for temp_file in temp_file_paths:
        try:
            with open(temp_file, 'r') as f:
                for line in f:
                    all_results.append(json.loads(line))
            os.remove(temp_file) # Clean up the temp file
        except FileNotFoundError:
            print(f"Warning: Temp file {temp_file} not found.", flush=True)

    # Sort results by the original index
    all_results.sort(key=lambda x: x['index'])

    # Write sorted results to the final file
    with open(save_path, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')

    # ✅ Removed logic for saving JSON cache (SQLite auto-persists)
    # --- Finalize ---
    try:
        os.rmdir(temp_dir) # Clean up the temp directory
    except OSError as e:
        print(f"Warning: Could not remove temp directory {temp_dir}. It may not be empty. Error: {e}", flush=True)

    end_time = time.time()
    exe_time = end_time - start_time
    print(f"\n🎉 Successfully saved {len(all_results)} results to {save_path}", flush=True)
    print(f"⏱️ Total time taken: {exe_time // 60:.0f}m, {exe_time % 60:.2f}s", flush=True)

    if args.do_eval:
        print("🔍 Evaluating results...")
        from evaluate import do_evaluate

        eval_time_start = time.time()
        do_evaluate(save_path)
        eval_time_end = time.time()
        eval_time = eval_time_end - eval_time_start
        print(f"Running len {args.end-args.begin} data, concurrency {args.concurrency}, search_budget {args.search_budget}, browse_budget {args.browse_budget}, n_query_results {args.n_query_results}")
        print(f"⏱️ Total running time: {exe_time // 60:.0f}m {exe_time % 60:.2f}s")
        print(f"⏱️ Avg time per sample (sec): {exe_time / len(all_data):.2f}s")
        print(f"⏱️ Evaluation time.  : {eval_time // 60:.0f}m {eval_time % 60:.2f}s")

        print(f"🔬 Analyzing results...")

        # from analyze_search import analyze_search_results
        from analyze import analyze_tool_results
        eval_path = save_path.replace(".jsonl", "_evalv1.jsonl")
        analysis_dict = analyze_tool_results(data_path=eval_path, total=len(all_data), budget_search=args.search_budget, budget_browse=args.browse_budget)
        # in minutes
        analysis_dict["time/execution"] = round(exe_time / 60, 3)
        analysis_dict["time/evaluation"] = round(eval_time / 60, 3)

        if args.wandb:
            wandb.init(project=args.wandb_project, name=wandb_name)
            wandb.config.update(args)
            wandb.log(analysis_dict)
            wandb.finish()
    else:
        print("Skipping evaluation as --do_eval is not set.")
    

if __name__ == "__main__":
    # Set the start method to 'spawn' at the very beginning.
    # 'force=True' is useful to prevent errors if the context
    # has already been set implicitly elsewhere.
    try:
        mp.set_start_method('spawn', force=True)
        print("--- Multiprocessing start method set to 'spawn' ---")
    except RuntimeError:
        # This can happen if the context is already set and cannot be changed.
        # It's good practice to handle this possibility.
        pass

    main()
