"""Chat-template + <think> prefill helper (§7 engineering constraints).

Constructs the full prompt string with a planted reasoning prefix inserted into
the model's <think> channel. Key rules:
  - Use apply_chat_template(..., add_generation_prompt=True, tokenize=False)
    then append <think>\n + prefill_text (if the template didn't already open <think>).
  - ALWAYS print the rendered template tail before generating — best habit to catch bugs.
  - Detect whether the template already opened <think>; only prepend if not.
  - DeepSeek-R1-Distill: no system prompt (§7).
  - Qwen3: enable_thinking=True; template ends without <think>, so we prepend it.
"""
from __future__ import annotations

from typing import Optional

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
PRINT_TAIL_CHARS = 200  # characters of the rendered prompt tail to print for debugging


def build_prefill_prompt(
    item: dict,
    prefill_text: str,
    tokenizer,
    *,
    system_prompt: Optional[str] = None,
    print_tail: bool = True,
) -> str:
    """Build the full prompt string with the prefill injected into <think>.

    Returns the raw string to pass to the tokenizer for encoding (tokenize=False
    in apply_chat_template means we get back a string, not token ids).

    The returned string ends with the prefill text (no closing </think> yet) so
    the model continues from there.
    """
    # Build the message list.
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    question = item["question"]
    choices = item["choices"]
    choice_lines = "\n".join(
        f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
    )
    user_content = f"{question}\n\n{choice_lines}"
    messages.append({"role": "user", "content": user_content})

    # Render via the model's chat template (no tokenization yet).
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        raise RuntimeError(f"apply_chat_template failed: {e}") from e

    # Detect whether the template already opened <think>.
    already_open = rendered.rstrip().endswith(_THINK_OPEN) or _THINK_OPEN in rendered.split(
        "assistant"
    )[-1]

    if already_open:
        prompt = rendered + "\n" + prefill_text
    else:
        prompt = rendered + _THINK_OPEN + "\n" + prefill_text

    if print_tail:
        tail = prompt[-PRINT_TAIL_CHARS:]
        print(f"\n[prefill tail — last {PRINT_TAIL_CHARS} chars]\n...{tail}\n")

    return prompt


def build_clean_prompt(
    item: dict,
    tokenizer,
    *,
    system_prompt: Optional[str] = None,
    print_tail: bool = False,
) -> str:
    """Build a clean prompt with no prefill (for baseline answer collection)."""
    return build_prefill_prompt(
        item,
        prefill_text="",
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        print_tail=print_tail,
    )
