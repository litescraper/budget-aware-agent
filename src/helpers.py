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

"""
Helper modules for infer_multiprocess
"""
import os
import re
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

TOOL_CALL_TAG = os.environ.get("TOOL_CALL_TAG", "tool_code")  # Default to 'tool_call' if not set


def remove_search_results(text: str) -> str:
    """Replace every <information>…</information> segment with <omitted>."""
    _INFO_RE = re.compile(r"<information>.*?</information>", flags=re.DOTALL)
    return _INFO_RE.sub("<omitted>", text)

def remove_tool_responses(text: str) -> str:
    """Replace every <tool_response>…</tool_response> segment with <omitted>."""
    _INFO_RE = re.compile(r"<tool_response>.*?</tool_response>", flags=re.DOTALL)
    return _INFO_RE.sub("<omitted>", text)

def is_good_query(text: str) -> bool:
    """
    Returns False if:
    - The query contains more than 4 quoted segments.
    - Any quoted segment contains more than 5 words.
    Otherwise, returns True.
    """
    # Find all quoted segments (e.g., "some phrase")
    quoted_phrases = re.findall(r'"(.*?)"', text)
    
    if len(quoted_phrases) > 2:
        return False
    
    for phrase in quoted_phrases:
        word_count = len(phrase.strip().split())
        if word_count > 5:
            return False
    
    return True

def find_later_tag(text: str):
    """
    Returns the tag that appears later in the text between <tool_call> and <answer>.
    If no such tag is found, returns None.
    """
    last_search = text.rfind(f"<{TOOL_CALL_TAG}>")
    last_answer = text.rfind("<answer>")

    if last_search == -1 and last_answer == -1:
        return None
    return f"<{TOOL_CALL_TAG}>" if last_search > last_answer else "<answer>"


def answer_is_none(text) -> bool:
    """
    Returns True if the answer is None, i.e., the text contains <answer>None</answer>.
    """
    if text is None:
        return True
    if "none" in text.lower():
        return True
    idk_list = ["i don't know", "i do not know", "i have no idea", "i am not sure", "i cannot answer"]
    return any(phrase in text.lower() for phrase in idk_list)

def extract_yesno_from_summary(text: str):
    m = re.search(r"<summary\b[^>]*>(.*?)</summary>", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None  # No summary tag found

    content = m.group(1).strip().lower()

    if "yes" in content:
        return True
    elif "no" in content:
        return False
    return None


def extract_yesno_from_verify(text: str):
    m = re.search(r"<verify\b[^>]*>(.*?)</verify>", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None  # No verify tag found

    content = m.group(1).strip().lower()

    if "yes" in content:
        return True
    elif "no" in content:
        return False
    return False

def extract_verify_old(text: str) -> str:
    """
    Extract content between the last [[verification]] and <verify> in the string.
    If <verify> is not found, extract content between [[verification]] and [[answer found]].
    If neither exists, return empty string.
    """
    if not isinstance(text, str):
        return ""
    
    # Prefer matching [[verification]] ... <verify>
    pattern_verify = re.compile(r"\[\[verification\]\](.*?)(?=<verify>)", re.DOTALL)
    matches_verify = pattern_verify.findall(text)
    if matches_verify:
        return matches_verify[-1].strip()
    
    # If <verify> is not found, match [[verification]] ... [[answer found]]
    pattern_answer_found = re.compile(r"\[\[verification\]\](.*?)(?=\[\[answer found\]\])", re.DOTALL)
    matches_answer = pattern_answer_found.findall(text)
    if matches_answer:
        return matches_answer[-1].strip()
    
    return ""

def extract_verify(text: str) -> str:
    """
    Extract content between the last [[verification]] and [[answer found]] in the string.
    If [[answer found]] is not found, extract content from the last [[verification]] to the next "[[".
    If the next "[[" does not exist, return content from the last [[verification]] to the end of the string.
    If none exists, return empty string.
    """
    if not isinstance(text, str):
        return "No verification found."

    # Find the last [[verification]]
    parts = text.split("[[verification]]")
    if len(parts) < 2:
        return "No verification found."

    segment = parts[-1]  # Part after the last verification

    # 1) Prefer to <verify>
    m = re.search(r"(.*?)(?=<verify>)", segment, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2) Then look for [[answer found]]
    m = re.search(r"(.*?)(?=\[\[answer found\]\])", segment, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3) Then look for the next [[
    idx_next = segment.find("[[")
    if idx_next != -1:
        return segment[:idx_next].strip()

    # 4) Otherwise to the end
    return segment.strip()



def extract_tool_call(text: str):
    """
    Extracts the content of the last <tool_call>...</tool_call> segment from the text.
    Returns as a json or None if no such segment is found.
    """
    if TOOL_CALL_TAG == "tool_call":
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    elif TOOL_CALL_TAG == "tool_code":  # Use the TOOL_CALL_TAG environment variable
        pattern = re.compile(r"<tool_code>(.*?)</tool_code>", re.DOTALL)
    else:
        raise ValueError(f"Unknown TOOL_CALL_TAG: {TOOL_CALL_TAG}")
    matches = pattern.findall(text)
    if not matches:
        return {}
    # Return the last match as a JSON object
    return json.loads(matches[-1])


def score_from_judge_response(eval_response: str) -> int:
    """Extract the yes/no verdict from an answer-evaluation judge response.

    Reads the explicit `correct:` field the judge prompt asks for, wherever it
    appears in the response. Judges that emit `correct:` before their reasoning
    would otherwise be scored off their final line, where reasoning such as
    "there are no meaningful differences" reads as a "no".
    """
    if not eval_response:
        return 0

    # Case-insensitive, tolerates markdown emphasis (**Correct**: Yes) and an
    # "incorrect:" phrasing (whose verdict is inverted); the lookbehind keeps
    # "correct_answer:" from matching.
    verdicts = re.findall(r"(?<![_\w])\*{0,2}(in)?correct\*{0,2}\s*:\s*\*{0,2}\s*(yes|no)\b",
                          eval_response, re.IGNORECASE)
    if verdicts:
        negated, verdict = verdicts[-1]
        is_yes = verdict.lower() == "yes"
        return 1 if is_yes != bool(negated) else 0

    last_line = eval_response.splitlines()[-1].strip().lower()
    if "yes" in last_line:
        return 1
    if "no" in last_line:
        return 0

    print("Evaluation response did not contain 'yes' or 'no'.")
    if "no" in eval_response.lower():
        return 0
    if "yes" in eval_response.lower():
        return 1
    return 0


def truncate(text: str, max_len: int = 150000) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else (text[:max_len] + "\n...[TRUNCATED]...")


def extract_json_from_string(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from a string containing a markdown code block
    and parses it into a Python dictionary.

    Args:
        text: The string containing the JSON code block to be extracted.

    Returns:
        A dictionary if a valid JSON block is found and parsed successfully,
        otherwise None.
    """
    # This regex pattern looks for a string that starts with ```json,
    # followed by any characters (non-greedy), and ends with ```.
    # It captures the content between the braces {}.
    # The re.DOTALL flag allows the '.' character to match newlines.
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)

    if match:
        # The captured JSON string is in the first group of the match
        json_string = match.group(1)
        try:
            # Parse the extracted string into a Python dictionary
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON. Details: {e}")
            return None
    else:
        print("Error: No JSON code block found in the string.")
        return None


def extract_decision_from_verify(text: str):
    # decision: "SUCCESS | CONTINUE | PIVOT"
    try:
        verify_json = extract_json_from_string(text)
        decision = verify_json.get("decision", "NONE")
    except:
        text = text.strip().lower()
        if '"success"' in text or "'success'" in text:
            decision = "SUCCESS"
        elif '"continue"' in text or "'continue'" in text:
            decision = "CONTINUE"
        elif '"pivot"' in text or "'pivot'" in text:
            decision = "PIVOT"
        else:
            decision = "NONE"
    return decision

def remove_verification_from_verify(text: str):
    # decision: "SUCCESS | CONTINUE | PIVOT"
    
    try:
        verify_json = extract_json_from_string(text)
        verify_json.pop("verification", None)
        verify_json.pop("justification", None)
        return verify_json
    except:
        verify_json = None
    return verify_json


def create_history(failed_answer_stat):
    if len(failed_answer_stat) == 0:
        return ""
    ret = ""
    cnt = 0
    for one in reversed(failed_answer_stat):
        if one['history'].strip() == "":
            continue
        ret += f"{one['history'].strip()}\n\n"
        cnt += 1
        if cnt >= 3:
            break
    if ret.strip() != "":
        ret = "Previous failed attempts:\n" + ret.strip() + "\n\n"

    return ret

import concurrent.futures
from functools import wraps

def timeout(seconds=60, default=None):
    """
    A decorator to apply a timeout to any synchronous function.
    
    Usage:
    @timeout(seconds=30, default="Timeout occurred")
    def my_slow_function():
        time.sleep(35)
        return "Done"
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use a ThreadPoolExecutor with a single worker
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Submit the function to run in the background thread
                future = executor.submit(func, *args, **kwargs)
                try:
                    # Wait for the result with a timeout
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    # If it times out, print a message and return the default value
                    pid = os.getpid()
                    print(f"[PID:{pid}] ⏰ Function '{func.__name__}' timed out after {seconds} seconds.", flush=True)
                    # We cancel the future to release resources if the task hasn't started
                    future.cancel()
                    return default
                except Exception as e:
                    pid = os.getpid()
                    print(f"[PID:{pid}] ❌ Function '{func.__name__}' raised an exception: {e}", flush=True)
                    return default
        return wrapper
    return decorator


# token calculation
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

class CacheHitEstimator:
    """
    Only estimate cache hit tokens count, ignore misses.
    - Group key = first group_prefix_len characters of input string
    - Each group keeps at most per_group_limit entries
    - Only do prefix matching within the same group
    """
    def __init__(
        self,
        model: str,
        group_prefix_len: int = 50,
        per_group_limit: int = 3,
        tokenizer=None,
    ):
        self.model = model
        self.tokenizer = tokenizer  # Store tokenizer for token counting
        self.group_prefix_len = group_prefix_len
        self.per_group_limit = per_group_limit
        self.groups: Dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self.per_group_limit)
        )

    @staticmethod
    def _lcp_prefix(a: str, b: str) -> str:
        """Return the longest common prefix of strings a and b."""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return a[:i]

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the appropriate method based on model type."""
        try:
            if self.tokenizer is not None:
                # Use a Hugging Face tokenizer when one is provided
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            else:
                # LiteLLM picks a model-appropriate tokenizer (tiktoken fallback)
                import litellm
                return int(litellm.token_counter(model=self.model, text=text))
        except Exception:
            # Fallback: character-based estimation
            return len(text) // 4

    def _group_key(self, s: str) -> str:
        return s[:self.group_prefix_len]

    def _same_group_candidates(self, s: str) -> List[str]:
        return list(self.groups[self._group_key(s)])

    def _update_group(self, s: str) -> None:
        self.groups[self._group_key(s)].append(s)

    def estimate_and_update(self, new_input: str) -> int:
        """
        Only return hit_tokens.
        """
        cands = self._same_group_candidates(new_input)
        if not cands:
            self._update_group(new_input)
            return 0

        best_lcp = ""
        for s in cands:
            lcp = self._lcp_prefix(new_input, s)
            if len(lcp) > len(best_lcp):
                best_lcp = lcp

        hit_tokens = self._count_tokens(best_lcp) if best_lcp else 0
        self._update_group(new_input)
        return hit_tokens

    def estimate_from_prev_segments(
        self,
        prev_input: str,
        prev_output: str,
        new_suffix: Optional[str] = None,
        update_history: bool = True
    ) -> int:
        """
        k+1 = prev_input + prev_output + new_suffix
        cache hit capped at len(prev_input) (output does not count as cache).
        """
        new_suffix = new_suffix or ""
        new_input = prev_input + prev_output + new_suffix

        cands = self._same_group_candidates(new_input)
        if not cands:
            if update_history:
                self._update_group(new_input)
            return 0

        best_lcp = ""
        for s in cands:
            lcp = self._lcp_prefix(new_input, s)
            if len(lcp) > len(best_lcp):
                best_lcp = lcp

        capped_lcp = best_lcp[:len(prev_input)]
        hit_tokens = self._count_tokens(capped_lcp) if capped_lcp else 0

        if update_history:
            self._update_group(new_input)
        return hit_tokens


def serialize_conversation_for_cache(conv: List[dict], si: str) -> str:
    """Convert conversation to stable text suitable as tokenizer input"""
    parts = []
    parts.append(f"<system>\n{si}\n</system>")
    for m in conv:
        role = m["role"]
        text = m["content"].strip()
        parts.append(f"<{role}>\n{text}\n</{role}>")
    return "\n".join(parts).strip()


def extract_plan_from_text(text: str):
    """
    Extracts the plan from the text.
    """
    pattern = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_the_last_plan(conversation: List[dict]):
    """
    Extracts the last plan from the conversation.
    """
    for m in reversed(conversation):
        if m["role"] == "model" and "<plan>" in m["content"]:
            return extract_plan_from_text(m["content"])
    return None