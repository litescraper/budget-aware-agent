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

    return prompt_v1.strip()


# ==============================
# ===== Agent Instructions =====
# ==============================

def pr_system_instruction_initial() -> str:

    # match with v8_2, change to tool, tool_response
    # v10_2 replace tool_call with tool_code
    prompt_v00_debug_d00 = """You are an AI reasoner with Google Search and Browsing tools. Solve the question by iterating: think, tool_code, tool_response, answer.

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

You should start with one or more cycles of (thinking about which tool to use -> performing tool code -> waiting for tool response), and end with (thinking about the answer -> answer of the question). The thinking processes, tool codes, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool codes, tool call parameters and tool response parameters.

## Step syntax

<think>
Thinking process. Analyze the query, your internal knowledge, and search results to build your reasoning.
</think>
<tool_code>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_code>
<tool_response>
tool_response here
</tool_response>

Repeat <think><tool_code> until you have the final answer.
<answer> Final solution only. </answer> 

## About answers

* Only write the final answer inside <answer> and </answer>.
* If you cannot find the answer, write <answer>None</answer>.
    """

    prompt_v00_debug_d1 = """You are an AI reasoner with Google Search and Browsing tools. Solve the question by iterating: think, tool_code, tool_response, answer.

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

You should start with one or more cycles of (thinking about which tool to use -> planning actions -> performing tool code -> waiting for tool response), and end with (thinking about the answer -> answer of the question). The thinking processes, tool codes, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, plans, tool codes, tool call parameters and tool response parameters.

## Step syntax

<think>
Thinking process. Analyze the query, your internal knowledge, and search results to build your reasoning.
</think>
<plan>
Planning process. The plan is a living checklist. Create and update a step by step plan for answering the question. Each step should define a single, actionable goal that can typically be accomplished with one tool call. Update the plan based on your thinking and tool responses ([ ] pending, [x] complete, [!] failed).
</plan>
<tool_code>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_code>
<tool_response>
tool_response here
</tool_response>

Repeat <think><plan><tool_code> until you have the final answer.
<answer> Final solution only. </answer> 

## About answers

* Only write the final answer inside <answer> and </answer>.
* If you cannot find the answer, write <answer>None</answer>.
    """
    
    prompt_v00_debug_d2 = """You are an AI reasoner with Google Search and Browsing tools. Solve the question by iterating: think, tool_code, tool_response, answer.

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

You should start with one or more cycles of (thinking about which tool to use -> performing tool code -> waiting for tool response), and end with (thinking about the answer -> answer of the question). The thinking processes, tool codes, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool codes, tool call parameters and tool response parameters.

## Step syntax

<think>
Thinking process. Analyze the query, your internal knowledge, and search results to build your reasoning. Create and update a step by step plan for answering the question. The plan is a living checklist. Each step should define a single, actionable goal that can typically be accomplished with one tool call. Update the plan based on your thinking and tool responses ([ ] pending, [x] complete, [!] failed).
</think>
<plan>
Planning process. The plan is a living checklist. Create and update a step by step plan for answering the question. Each step should define a single, actionable goal that can typically be accomplished with one tool call. Update the plan based on your thinking and tool responses ([ ] pending, [x] complete, [!] failed).
</plan>
<tool_code>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_code>
<tool_response>
tool_response here
</tool_response>

Repeat <think><tool_code> until you have the final answer.
<answer> Final solution only. </answer> 

## About answers

* Only write the final answer inside <answer> and </answer>.
* If you cannot find the answer, write <answer>None</answer>.
    """
    
    return prompt_v00_debug_d00.strip()



# ==========================
# ===== Answer Module ======
# ==========================
# <answer>

def pr_system_instruction_generate_final_answer() -> str:

    prompt_v3 = """
Identify the final answer from the question and response below, and output it wrapped in <answer> and </answer> tags. For example: <answer>Beijing</answer>. If no answer is found, output <answer>None</answer>. Do not include any explanation."""

    return prompt_v3.strip()


def pr_prompt_generate_final_answer() -> str:

    prompt_v3 = """
Directly provide the final enclosed in <answer> and </answer>."""

    return prompt_v3.strip()

# ==========================
# ===== Browsing Module ====
# ==========================
# browsing only, no tag
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


    """
    This instruction is used to continue the generation of the agent's response when the agent has a none answer but the search budget is not exhausted.
    """
    prompt_v10_2 = """The answer remains unclear so far, but the tool budget allows for further exploration. I'll now try a different approach to continue making progress. I'll still follow the steps of <think><tool_code> and <answer>."""
    return prompt_v10_2.strip()


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

    return prompt_v1.strip()