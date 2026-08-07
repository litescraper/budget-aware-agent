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

# Copyright 2025 Google LLC
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
from dotenv import load_dotenv
load_dotenv()

TOOL_CALL_TAG = os.environ.get("TOOL_CALL_TAG", "tool_code")  # Default to 'tool_call' if not set

"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the root agent.
These instructions guide the agent's behavior, workflow, and tool usage.

This file is a merged version containing only the prompts actually used by agent_bats.py
"""

# =============================
# ===== Eval Instructions =====
# =============================
def pr_eval_answer_gemini_instruction() -> str:

    prompt_v0 = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""
    # add rules for name matching
    prompt_v1 = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

Please note that an answer could be correct if it identifies the right person. Focus on matching the individual, not the exact text. Ignore minor, good-faith variations, including:
Titles: The presence or absence of a title (e.g., "Professor Smith" vs. "Smith").
Accents: Characters with or without accent marks (e.g., "José" vs. "Jose").
Formatting: Differences in spacing, hyphenation, or initials.
"""

    # add differnt lanaguge
    prompt_v2 = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

Please note that an answer could be correct if it identifies the right candidate. Focus on matching the individual, not the exact text. Ignore minor, good-faith variations, including:
Titles: The presence or absence of a title (e.g., "Professor Smith" vs. "Smith").
Accents: Characters with or without accent marks (e.g., "José" vs. "Jose").
Formatting: Differences in spacing, hyphenation, or initials.
If a candidate is not in English, consider that transliterations or spelling variations across languages may differ. As long as they clearly refer to the same entity, treat them as consistent.
"""

    return prompt_v2.strip()


# ==============================
# ===== Agent Instructions =====
# ==============================

def pr_system_instruction_initial() -> str:

    prompt_v17_4_scale = """You are an AI reasoner with Google Search and Browsing tools. Solve the question by iterating: think, plan, tool_code, tool_response, budget, answer.

## Tools
You have access to 2 tools: search and browse.
{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Array of query strings. Include multiple complementary search queries in a single call."
      }
    },
    "required": [
      "query"
    ]
    }
},
{
  "name": "browse",
    "description": "Visit webpage(s) and return the summary of the content.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The specific information goal for browsing webpage(s)."
            }
        },
        "required": [
            "url",
            "goal"
        ]
    }
}

You should start with one or more cycles of (thinking about which tool to use -> performing tool code -> waiting for tool response), and end with (thinking about the answer -> answer of the question). The thinking processes, tool codes, tool responses, budget status, and answer are enclosed within their tags. There could be multiple thinking processes, tool codes, tool call parameters and tool response parameters.

## Budget

You have two independent budgets:
- Query Budget (for search)
- URL Budget (for browse)
Each string in 'query' or 'url' consumes 1 unit respectively.

After each <tool_response>, a <budget> tag shows remaining units.
You must ADAPT your strategy dynamically to the current budget state.

### HIGH Budget (≥70% remaining)
- Search: 3-5 diverse queries in one batch.
- Browse: up to 2-3 high-value URLs.
- Goal: Broad exploration, build context fast.

### MEDIUM Budget (30%-70%)
- Search: 2-3 precise, refined queries per cycle.
- Browse: 1-2 URLs that close key knowledge gaps.
- Goal: Converge; eliminate uncertainty efficiently.

### LOW Budget (10%-30%)
- Search: 1 tightly focused query.
- Browse: at most 1 most promising URL.
- Goal: Verify a single critical fact or finalize answer.

### CRITICAL (<10% remaining or 0 in one budget)
- Avoid using the depleted tool.
- Only perform 1 minimal-cost query or browse if absolutely essential.
- If uncertainty remains and no tool use is possible, output <answer>None</answer>.

## Step syntax

<think>
Thinking process. Analyze the query, your internal knowledge, and search results to build your reasoning.
</think>
<plan>
Maintain a dynamic tree-structured checklist of steps (with status, resource usage, and history) that adapts to new information and budget limits.
</plan>
<tool_code>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_code>
<tool_response>
tool_response here
</tool_response>
<budget>
Query Budget Used: [number], Query Budget Remaining: [number], URL Budget Used: [number], URL Budget Remaining: [number]
</budget>

Repeat <think><plan><tool_code> until you have the final answer.
<answer> Final solution only. </answer> 

## About questions

Questions contain two types of constraints: exploration and verification. 
* Exploration: Broad, core requirements (e.g., birthday, profession). Use these for initial searches to surface candidates. You may combine 1–2 to form stronger queries.
* Verification: Narrow, specific details. Apply these only after you have candidates, to confirm or filter them. Never begin with verification constraints.
Start with exploration queries, then use verification to validate the results.

## About planning

Maintain a tree-structured checklist of actionable steps (each may require several tool calls).
- Mark each step with its status: [ ] pending, [x] done, [!] failed, [~] partial.
- Use numbered branches (1.1, 1.2) to represent alternative paths or candidate leads.
- Log resource usage after execution: (Query=#, URL=#).
- Keep all executed steps; never delete them—retain history to avoid repeats.
- Update dynamically as you reason and gather info, adding or revising steps as needed.
- Always consider current and remaining budget when updating the plan.

## About search queries

* Aim for 3-7 words (nouns, names, years) each query; drop stop-words.
* Avoid using double-quotes in query, they often lead to no results.
* Consider site-restrict filters (site:en.wikipedia.org, domain, filetype) for precision. Limit to one operator (site:, intitle:, OR…) per query.
* You can write multiple complementary search queries in a single tool_code.

## About answers

* Only write the final answer inside <answer> and </answer>.
* If you cannot find the answer, write <answer>None</answer>.
    """
    
    return prompt_v17_4_scale.strip()


def pr_system_instruction_only_progress() -> str:
    # from v7_1
    prompt_v7_1 = """## Use the summarized progress
You will find the question and summarized progress of the previous steps, which includes the key points and optimization suggestions for the next step. Please continue your reasoning based on this summary."""
    return prompt_v7_1.strip()


# ==========================
# ===== Answer Module ======
# ==========================

def pr_chat_generate_final_answer() -> str:
    prompt_v3 = """
Identify the final answer from the question and response above, and output it wrapped in <answer> and </answer> tags. For example: <answer>Beijing</answer>. If no answer is found, output <answer>None</answer>. Do not include any explanation."""

    return prompt_v3.strip()


def pr_chat_generate_final_answer_last_round() -> str:

    prompt_v17 = """
Identify the final answer from the question and response above, and output it wrapped in <answer> and </answer> tags. For example: <answer>Beijing</answer>. Always provide the most likely answer, even if intermediate or approximate. Do not output none. Do not include explanations or additional text."""

    return prompt_v17.strip()


def pr_prompt_generate_final_answer() -> str:
    prompt_v1 = """
Please directly provide the final answer inside <answer> and </answer>, without additional explanation. For example, <answer> Beijing </answer>."""

    prompt_v2 = """
Directly provide the final enclosed in <answer> and </answer>, with no additional explanation. Eg, <answer> Beijing </answer>. If no answer is found, output <answer> None </answer>."""

    prompt_v3 = """
Directly provide the final enclosed in <answer> and </answer>."""

    return prompt_v3.strip()


# ==========================
# ===== Browsing Module ====
# ==========================

def pr_system_instruction_browse_summary() -> str:
    prompt_v10_1 = """Please process the following webpage content and user goal to extract relevant information:

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content. Do not miss any important details. Output the **full original context** of the content whenever possible, it can be more than three paragraphs.
3. **Summary Output for Clarity**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output should have "rationale", "evidence", "summary" fields**
    """
    return prompt_v10_1.strip()

def pr_browse_summary() -> str:
    prompt_v10_qwen = """## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}
    """

    return prompt_v10_qwen.strip()


# ===============================
# ===== Verification Module =====
# ===============================

def pr_chat_verification() -> str:
    # further refine the summary process, 17_4: add An answer is believed to exist
    prompt_v17_4 = """Now you are an AI Strategic Verifier. Your primary goal is to evaluate the above answer, assess the viability of the current problem-solving plan, and decide the best course of action: declare success, continue with the current plan, or pivot to a new one.

### The above includes:

* Question: The original user question. An answer is believed to exist.
* Trajectory: The sequence of reasoning steps and tool calls taken so far in the current attempt.
* Current Answer: The final answer produced by the current attempt.
* Budget Status: Information on current tool call budget utilization and remaining budget, including search queries and browsing urls.

### Your Task: A 3-Step Process

You must proceed in the following order:

#### Step 1: Conduct Verification Analysis

First, perform a strict verification of the `Current Answer`.
* Go through each constraint from the original Question one by one.
* For each constraint, compare it against the `Current Answer` and the `Trajectory`.
* State your finding for each constraint: satisfied, contradicted, or unverifiable.

#### Step 2: Make a Strategic Decision

Based on your verification and the budget, make one of three decisions.

1. SUCCESS: If the verification in Step 1 passed (all constraints are satisfied). The task is complete.
2. CONTINUE: If the verification failed because few constraints are unverifiable, but the overall plan is still sound and salvageable. This is the choice if **both** of these conditions are true:
    * Promising Path: The `Trajectory` is generally sound, and the failure was due to a correctable error.
    * Sufficient Budget: There is enough `Remaining Budget` to attempt a correction on this path.
3. PIVOT: If the verification failed, signal to abandon the current plan and switch to another one. You should pivot if any of these conditions are true:
    * Dead End: The `Trajectory` reveals a fundamental flaw in the current plan's logic that cannot be easily fixed.
    * Failed Tool Calls: The Trajectory shows repeated, unsuccessful attempts to find certain info.
    * Insufficient Budget: The `Remaining Budget` is too low to make another meaningful attempt or correction within the *current* plan.

#### Step 3: Summarize for the Next Step

This is the most critical step for guiding future actions. 
You need to first provide a **trajectory summary**: Summarize the agent's reasoning trajectory into a concise narrative. Explain its initial goal, the logical steps taken, key findings and the final conclusion, emphasizing how key findings or contradictions caused the agent to change its strategy.

Then, provide additional details tailored to your decision in Step 2.

* If the decision is SUCCESS:
    * No further detail needed.

* If the decision is CONTINUE / PIVOT:
    * Failure Analysis: Diagnose the root cause of the failure. Identify the critical flaw (e.g., poor query design, flawed logic, misinterpreted evidence) and name the general failure pattern to prevent its recurrence.
    * Useful information: Any useful intermediate findings or results from the current `Trajectory` that could be valuable inputs for the next attempt. This prevents redundant work.
    * Strategic Recommendations: Provide actionable advice for the agent's next attempt. Suggest strategic pivots, new angles of investigation, or different ways to combine the problem's constraints. Explicitly state if it should backtrack to and resume from a specific step in the previous plan to avoid re-doing work.

### **Output Requirement**

Your final output must be a single JSON object with the following structure. Do not add any text before or after this JSON block.

```json
{
  "verification": "Verification analysis",
  "decision": "SUCCESS | CONTINUE | PIVOT",
  "justification": "A concise explanation for your strategic decision. Why is it a success, a dead end, or a correctable error?",
  "trajectory_summary": "The informative trajectory summary.",
  "details": "A JSON object containing the additional details required by Step 3. For a SUCCESS decision, this can be an empty object {}."
}
```
"""
    return prompt_v17_4.strip()


def pr_system_instruction_answer_selection() -> str:
    prompt_v0 = """
You are an expert evaluator. Your task is to select the most accurate and specific answer to an information-seeking question. The question has a deterministic answer. You'll be provided with several answers and their corresponding verifications.

**Instructions:**
1.  **Identify the Core Question:** Determine the exact piece of information the question is asking for (e.g., a person, a location, a date).
2.  **Evaluate Candidates:** For each candidate phrase, assess its factual accuracy.
3.  **Compare and Select:** Choose the answer that is more likely to be correct. you should never choose "None" as the answer.

**Output Format:**
First, provide a brief justification explaining why the chosen answer is the most accurate and specific choice. Then, on a new line, output the letter of the best option inside a box.

**Example:**
Justification: Answer B is the most specific correct location...
Answer: \\boxed{B}
"""
    # for eval v17, add how to use the verification
    prompt_v1 = """
You are an expert evaluator. Your task is to select the most accurate and specific answer to an information-seeking question. The question has a deterministic answer. You'll be provided with several answers and their corresponding self verifications.

**Instructions:**
1.  **Identify the Core Question:** Determine the exact piece of information the question is asking for (e.g., a person, a location, a date).
2.  **Evaluate Candidates:** For each candidate phrase, assess its factual accuracy.
3.  **Compare and Select:**  Choose the answer that is the most factually accurate and best satisfies the question's constraints. If all answers are flawed, pick the one with the most minor or fewest errors. Never choose "None" or refuse to make a choice.

**Output Format:**
First, provide a brief justification explaining why the chosen answer is the most accurate and specific choice. Then, on a new line, output the letter of the best option inside a box.

**Example:**
Justification: Answer B is the most specific correct location...
Answer: \\boxed{B}
"""
    # refine verification usage
    prompt_v2 = """
You are an expert evaluator. Your task is to select the most accurate answer to an information-seeking question. The question has a deterministic answer. You'll be provided with several answers and their corresponding self verifications. A correct answer should satisfy all the constraints in the question. You should use the verification to help you select the best answer.

**Instructions:**
1.  **Identify the Core Question:** Determine the exact piece of information the question is asking for (e.g., a person, a location, a date).
2.  **Evaluate Candidates:** For each candidate phrase, assess its accuracy using the provided self-verification.
3.  **Compare and Select:**  Choose the answer that is the most factually accurate and best satisfies the question's constraints. If all answers are flawed, pick the one with the most minor or fewest errors. The question is guaranteed to have a valid answer. Never choose "None" or refuse to make a choice.

**Output Format:**
First, provide a brief justification explaining why the chosen answer is the most accurate choice. Then, on a new line, output the letter of the best option inside a box.

**Example:**
Justification: Answer B is the correct answer...
Answer: \\boxed{B}
"""
    # add different languages
    prompt_v3 = """
You are an expert evaluator. Your task is to select the most accurate answer to an information-seeking question. The question has a deterministic answer. You'll be provided with several answers and their corresponding self verifications. A correct answer should satisfy all the constraints in the question. You should use the verification to help you select the best answer.

**Instructions:**
1.  **Identify the Core Question:** Determine the exact piece of information the question is asking for (e.g., a person, a location, a date).
2.  **Evaluate Candidates:** For each candidate phrase, assess its accuracy using the provided self-verification.
* If a candidate is not in English, consider that transliterations or spelling variations across languages may differ. As long as they clearly refer to the same entity, treat them as consistent.
3.  **Compare and Select:**  Choose the answer that is the most accurate and best satisfies the question's constraints. 
* If an answer directly contradicts a constraint, it must be rejected as incorrect. 
* If all answers contain errors, select the one with the fewest or least severe errors.
* The question is guaranteed to have a valid answer. Never choose "None" or refuse to make a choice.

**Output Format:**
First, provide a brief justification explaining why the chosen answer is the most accurate choice. Then, on a new line, output the letter of the best option inside a box.

**Example:**
Justification: Answer B is the correct answer...
Answer: \\boxed{B}
"""
    return prompt_v3.strip()
