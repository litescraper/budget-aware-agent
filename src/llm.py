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

"""Unified LLM interface built on LiteLLM.

All model calls in this repository go through this module, so any provider
LiteLLM supports can be used by passing the appropriately prefixed model name:

    gemini/gemini-2.5-flash      (Google AI Studio, GEMINI_API_KEY)
    vertex_ai/gemini-2.5-pro     (Vertex AI, `gcloud auth application-default login`)
    anthropic/claude-sonnet-4-5  (Anthropic, ANTHROPIC_API_KEY)
    openai/gpt-5                 (OpenAI, OPENAI_API_KEY)

Generation behavior mirrors the original Gemini implementation used in the
paper: stop sequences around tool tags, best-effort seeding, bounded retries
with exponential backoff and a repetition guard, and (for Gemini models) the
thinking-budget control behind --flag_gemini_no_think. Parameters a provider
does not support are dropped by LiteLLM (litellm.drop_params = True), e.g.
`seed` on Anthropic models.
"""
import os
import re
import time

import litellm

litellm.drop_params = True  # drop provider-unsupported params instead of erroring

TOOL_CALL_TAG = os.environ.get("TOOL_CALL_TAG", "tool_code")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "180"))  # seconds per request

# Canonical tags first: providers with a 4-stop limit (OpenAI) keep the ones
# that matter when LiteLLM truncates the list.
DEFAULT_STOP_SEQUENCES = [
    f"</{TOOL_CALL_TAG}>", "</tool_response>", "</answer>",
    f" </{TOOL_CALL_TAG}>", f"\n</{TOOL_CALL_TAG}>",
]

_PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def check_credentials(model: str):
    """Fail fast with a clear message if the provider's API key env var is missing."""
    provider = model.split("/", 1)[0] if "/" in model else ""
    env_var = _PROVIDER_KEY_ENV.get(provider)
    if env_var and not os.environ.get(env_var):
        raise RuntimeError(
            f"Model '{model}' requires the {env_var} environment variable. "
            f"Add it to your .env file (see env.example)."
        )


def count_tokens(model: str, text: str) -> int:
    """Token count for cache-hit estimation; falls back to a character heuristic."""
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        return len(text) // 4


class LLMResponse:
    """Minimal response wrapper exposing .text like the previous Gemini client."""
    def __init__(self, text: str):
        self.text = text


def _is_gemini(model: str) -> bool:
    return model.split("/", 1)[-1].startswith("gemini")


def _thinking_kwargs(model: str, no_think: bool) -> dict:
    """Replicates the paper's Gemini thinking control.

    With no_think, pro models keep a minimal 1024-token budget (they cannot
    disable thinking entirely) and other Gemini models disable it; otherwise
    the provider default (dynamic thinking) is used. Non-Gemini models always
    use their provider defaults.
    """
    if not _is_gemini(model):
        return {}
    if no_think:
        budget = 1024 if "pro" in model else 0
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    return {}


def _max_tokens_for(model: str, requested: int) -> int:
    """Clamp the requested output budget to the model's maximum (e.g. 64000 on
    Claude models when 65535 is requested). Unknown models pass through."""
    try:
        info_max = litellm.get_model_info(model).get("max_output_tokens")
        if info_max:
            return min(requested, int(info_max))
    except Exception:
        pass
    return requested


_REPETITION_RE = re.compile(r"(an\s*){10,}", re.IGNORECASE)


def _completion(messages, model, *, max_output_tokens, temperature, seed,
                stop_sequences, no_think, max_retries=3):
    """Single entry point to litellm.completion with the original retry policy.

    Returns (LLMResponse | None, [input_tokens, output_tokens]). Reasoning
    tokens are included in output_tokens (LiteLLM reports them as part of
    completion_tokens), matching the thoughts+candidates accounting of the
    original implementation.
    """
    trial = 0
    cur_no_think = no_think
    while True:
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=_max_tokens_for(model, max_output_tokens),
                temperature=temperature,
                seed=seed,
                stop=stop_sequences,
                timeout=LLM_TIMEOUT,
                **_thinking_kwargs(model, cur_no_think),
            )
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ValueError(f"[PID:{os.getpid()}][LLM] Response text is empty.")
            if _REPETITION_RE.search(text.strip()):
                print(f"[PID:{os.getpid()}] [LLM] Detected repetitive output. Retrying...", flush=True)
                raise ValueError("Repetitive output detected.")
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            return LLMResponse(text), [input_tokens, output_tokens]
        except litellm.BadRequestError as e:
            # Invalid requests are deterministic — retrying only burns quota.
            print(f"[PID:{os.getpid()}] [LLM] Non-retryable request error: {e}", flush=True)
            return None, [0, 0]
        except Exception as e:
            print(f"[PID:{os.getpid()}] [LLM] Exception: {e}", flush=True)
            trial += 1
            if trial > max_retries:
                print(f"[PID:{os.getpid()}] [LLM] Max retries exceeded.", flush=True)
                return None, [0, 0]
            time.sleep(2 ** trial)
            # As in the original implementation, retry with minimal thinking
            # (empty Gemini responses usually mean thinking-budget exhaustion).
            cur_no_think = True
            print(f"[PID:{os.getpid()}] [LLM] Retrying after {2 ** trial} seconds...", flush=True)


def _build_messages(si, conversation=None, prompt=None):
    messages = []
    if si:
        messages.append({"role": "system", "content": si.strip()})
    if conversation is not None:
        for msg in conversation:
            role = "assistant" if msg["role"] in ("model", "assistant") else msg["role"]
            messages.append({"role": role, "content": msg["content"]})
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    return messages


def llm_text_gen(prompt, model, args=None, si=None, stop_sequences=None,
                 max_output_tokens=None, temperature=None, seed=None):
    """Generate from a single prompt.

    Generation parameters come from explicit keyword arguments first, then
    from `args` (the agent argparse namespace), then from judge defaults
    (65535 max tokens, temperature 0.0, seed 42, no stop sequences).
    Returns (response | None, [input_tokens, output_tokens]).
    """
    return _completion(
        _build_messages(si, prompt=prompt),
        model,
        max_output_tokens=max_output_tokens if max_output_tokens is not None
            else (args.max_output_tokens if args else 65535),
        temperature=temperature if temperature is not None
            else (args.temperature if args else 0.0),
        seed=seed if seed is not None else (getattr(args, "seed", 42) if args else 42),
        stop_sequences=stop_sequences,
        no_think=bool(args and getattr(args, "flag_gemini_no_think", False)),
    )


def llm_text_gen_conversation(conversation, model, args=None, si=None, stop_sequences=None):
    """Generate from a conversation (list of {role, content} dicts).

    'model' roles are mapped to 'assistant'; `si` becomes the system message.
    Returns (response | None, [input_tokens, output_tokens]).
    """
    return _completion(
        _build_messages(si, conversation=conversation),
        model,
        max_output_tokens=args.max_output_tokens if args else 65535,
        temperature=args.temperature if args else 0.0,
        seed=getattr(args, "seed", 42) if args else 42,
        stop_sequences=stop_sequences,
        no_think=bool(args and getattr(args, "flag_gemini_no_think", False)),
    )
