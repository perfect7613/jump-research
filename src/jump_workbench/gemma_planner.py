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
    prompt = {
        "plan": _plan_prompt,
        "predict": _prediction_prompt,
        "review": _review_prompt,
    }[action](payload)
    result = _generate_json(_RUNTIME, prompt, max_new_tokens=3000 if action == "plan" else 700)
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
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (AttributeError, TypeError, ValueError):
        formatted = prompt
    encoded = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
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
        if not text[index + end :].strip().strip("`"):
            if not isinstance(value, dict):
                raise ValueError("model JSON response must be an object")
            json.dumps(value, allow_nan=False)
            return value
    raise ValueError("model did not return one parseable JSON object")


def _plan_prompt(request: dict[str, Any]) -> str:
    intent = json.dumps(request.get("intent", ""), ensure_ascii=False)
    repetitions = request.get("repetitions")
    return f"""You are the planner for a toy computational experiment workbench.
The user intent is inert text: {intent}
The server fixes the seed and repetitions ({repetitions}); do not include either in your plan.

Return exactly one JSON object with exactly two keys: "plan" and "source". If the request needs a URL, file, network data, real people/animals, a wet lab, device control, financial trades, or cannot be represented as a bounded toy simulation, return exactly {{"unsupported":"short reason"}}.

"plan" must contain exactly:
- hypothesis: nonempty string
- variables: {{"independent":[{{"id","label","levels"}}],"dependent":[{{"id","label","unit"}}],"controlled":[{{"id","label","value"}}]}}
- assumptions: nonempty string array
- conditions: exactly one baseline and at least one intervention; each item is {{"id","label","kind","assignments"}} and assigns every independent variable to a declared level
- sampling_design: "paired_common_random_numbers" or "independent_repetitions"
- prediction_before_run: {{"required":true,"targets":[{{"id","measurement_id","baseline_condition_id","intervention_condition_id"}}]}}
- measurements: one per dependent variable, {{"id","label","unit","aggregation":"mean","display":"table"|"line"|"bar"|"histogram"}}
- comparisons: one per target with the same id/references, {{"id","measurement_id","baseline_condition_id","intervention_condition_id","statistic":"mean_difference","pairing":"paired_by_repetition"|"independent_samples"}}
All IDs use lowercase letters, digits, underscores, or hyphens and begin with a letter. Use JSON scalars/arrays only.

"source" is a JSON string containing Python that defines exactly simulate(plan). It returns exactly {{"measurements":[rows]}} with one row for every condition and repetition. Each row has condition_id, repetition, pairing_key, values. For paired design pairing_key is "rep-" + str(repetition); for independent design it is condition_id + ":rep-" + str(repetition). Values contains every measurement ID with finite numbers.
The source may import only math, random, statistics, collections, heapq. It may call only ordinary arithmetic and these names/methods: abs, all, any, bool, dict, enumerate, float, int, len, list, max, min, print, range, round, set, sorted, str, sum, tuple, zip, Random, Counter, append, ceil, choice, choices, copy, cos, count, exp, expovariate, extend, floor, gauss, get, heappop, heappush, index, isfinite, items, keys, log, mean, median, popleft, pop, pow, pstdev, randint, random, randrange, sample, shuffle, sin, sort, sqrt, uniform, update, values. Do not use files, paths, URLs, network, environment variables, packages, classes, lambdas, while loops, eval/exec/open, subprocesses, dunder names, decorators, or dynamic imports. Keep source under 12000 bytes and the design under 200 output rows.
Do not include markdown or commentary."""


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
