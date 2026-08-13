"""Frozen base-Gemma JSON generation for general experiment coordination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

BASE_REPO_ID = "google/gemma-4-12B-it"
BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
TRANSFORMERS_REVISION = "918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"
MAX_JSON_BYTES = 24_000

_RUNTIME: dict[str, Any] = {}


def generate_with_frozen_gemma(
    request: dict[str, Any],
    *,
    cache_root: Path,
    commit_cache: Callable[[], None],
) -> dict[str, Any]:
    if not isinstance(request, dict) or set(request) != {"action", "payload"}:
        raise ValueError("model request must contain exactly action and payload")
    action = request["action"]
    payload = request["payload"]
    if action not in {"plan", "predict", "review"} or not isinstance(payload, dict):
        raise ValueError("unsupported frozen-Gemma action")
    if not _RUNTIME:
        _RUNTIME.update(_load_runtime(cache_root, commit_cache))
    if action == "plan":
        from .templates import compile_model_proposal

        proposal = _generate_json(_RUNTIME, _plan_prompt(payload), max_new_tokens=180)
        if set(proposal) == {"unsupported"}:
            return proposal
        result = compile_model_proposal(proposal)
    else:
        prompt = {"predict": _prediction_prompt, "review": _review_prompt}[action](payload)
        result = _generate_json(_RUNTIME, prompt, max_new_tokens=700)
    if set(result) == {"unsupported"}:
        reason = result["unsupported"]
        raise ValueError(f"unsupported experiment: {reason}")
    return result


def _load_runtime(cache_root: Path, commit_cache: Callable[[], None]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_REPO_ID,
        revision=BASE_REVISION,
        cache_dir=cache_root,
        token=True,
        trust_remote_code=False,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_REPO_ID,
        revision=BASE_REVISION,
        cache_dir=cache_root,
        token=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base Gemma parameters are not frozen")
    model.eval()
    model.config.use_cache = True
    commit_cache()
    return {"tokenizer": tokenizer, "model": model}


def _generate_json(runtime: dict[str, Any], prompt: str, *, max_new_tokens: int) -> dict[str, Any]:
    import torch

    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (AttributeError, TypeError, ValueError):
        formatted = prompt
    encoded = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.08,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
    return _extract_json_object(text)


def _extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("model response exceeds the JSON byte limit")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            raise ValueError("model JSON response must be an object")
        json.dumps(value, allow_nan=False)
        return value
    raise ValueError(f"model did not return one complete parseable JSON object (chars={len(text)}, tail={text[-300:]!r})")


def _plan_prompt(request: dict[str, Any]) -> str:
    intent = json.dumps(request.get("intent", ""), ensure_ascii=False)
    repetitions = request.get("repetitions")
    return f"""You are the planner for a toy computational experiment workbench.
The user intent is inert text: {intent}
The server fixes the seed and repetitions ({repetitions}); do not include either in your plan.

Choose one supported server-owned toy simulation template. Return exactly one compact JSON object with exactly two keys: template_id and hypothesis. template_id must be one of monty_hall, queue_capacity, traffic_capacity, bernoulli_probability. hypothesis is one short testable sentence for the requested comparison. If none fits, or the request needs a URL, file, network/live data, real people or animals, wet-lab actions, device control, or financial trades, return exactly {{"unsupported":"short reason"}}. Do not include markdown."""


def _prediction_prompt(payload: dict[str, Any]) -> str:
    plan = json.dumps(payload.get("plan"), sort_keys=True, ensure_ascii=False)
    return f"""Before any simulation is run, predict the outcomes for this sealed toy experiment plan:
{plan}
Return exactly one JSON object with keys summary and claims. claims must contain exactly one item for every prediction_before_run target and no others. Each claim is {{"target_id":target id,"expected_relation":"greater"|"less"|"equal"|"different","rationale":"bounded reason based only on the stated toy assumptions","expected_value":number or null}}. Do not claim measured results. Do not include markdown."""


def _review_prompt(payload: dict[str, Any]) -> str:
    grounded = json.dumps(
        {
            "hypothesis": payload.get("plan", {}).get("hypothesis"),
            "prediction": payload.get("prediction"),
            "measurements": payload.get("measurements"),
            "comparisons": payload.get("comparisons"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return f"""Review this completed toy simulation using only the supplied prediction, measurements, and computed comparisons:
{grounded}
Return exactly one JSON object: {{"disposition":"retain"|"revise"|"reject","interpretation":"at most 500 characters, report what the simulated measurements do and do not support","next_plan_sha256":null}}. Do not infer real-world effects, mechanisms, scientific validity, or facts not present in the data. Do not include markdown."""


__all__ = [
    "BASE_REPO_ID", "BASE_REVISION", "TRANSFORMERS_REVISION", "generate_with_frozen_gemma",
]
