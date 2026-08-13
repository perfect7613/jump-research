"""Fail-closed T0--T4 sentinel and token-boundary resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .capture import Timepoint


class TokenizerLike(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SentinelDefinition:
    timepoint: Timepoint
    text: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("sentinel text must be nonempty")
        if not self.token_ids or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.token_ids):
            raise ValueError("sentinel token_ids must be nonempty nonnegative integers")


@dataclass(frozen=True)
class FollowerOption:
    text: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("immediate follower text must be nonempty")
        if not self.token_ids or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.token_ids):
            raise ValueError("immediate follower token_ids must be nonempty nonnegative integers")


@dataclass(frozen=True)
class ImmediateFollowerRule:
    timepoint: Timepoint
    kind: str
    options: tuple[FollowerOption, ...]

    def __post_init__(self) -> None:
        expected = {Timepoint.T3: "adequacy_boolean", Timepoint.T4: "replacement_law_tuple"}
        if self.timepoint not in expected or self.kind != expected[self.timepoint]:
            raise ValueError("immediate follower rules must bind T3 to adequacy_boolean and T4 to replacement_law_tuple")
        if (
            not isinstance(self.options, tuple)
            or not self.options
            or any(not isinstance(option, FollowerOption) for option in self.options)
        ):
            raise ValueError("immediate follower rule options must be a nonempty FollowerOption tuple")
        texts = [option.text for option in self.options]
        if len(texts) != len(set(texts)) or any(
            left != right and right.startswith(left) for left in texts for right in texts
        ):
            raise ValueError("immediate follower option texts must be unique and non-overlapping")


@dataclass(frozen=True)
class BoundaryManifest:
    tokenizer_id: str
    tokenizer_revision: str
    sentinels: tuple[SentinelDefinition, ...]
    immediate_followers: tuple[ImmediateFollowerRule, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoundaryManifest":
        required = {"tokenizer_id", "tokenizer_revision", "sentinels", "immediate_followers"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"boundary manifest must contain exactly {sorted(required)}")
        definitions = []
        raw = value["sentinels"]
        if not isinstance(raw, Mapping) or set(raw) != {point.value for point in Timepoint}:
            raise ValueError("boundary manifest must define exactly T0--T4")
        for point in Timepoint:
            item = raw[point.value]
            if not isinstance(item, Mapping) or set(item) != {"text", "token_ids"}:
                raise ValueError(f"sentinel {point.value} must contain exactly text and token_ids")
            token_ids = item["token_ids"]
            if not isinstance(token_ids, list):
                raise ValueError(f"sentinel {point.value}.token_ids must be an array")
            definitions.append(SentinelDefinition(point, item["text"], tuple(token_ids)))
        follower_rules = []
        followers = value["immediate_followers"]
        if not isinstance(followers, Mapping) or set(followers) != {"T3", "T4"}:
            raise ValueError("boundary manifest must define immediate follower rules for exactly T3 and T4")
        for point in (Timepoint.T3, Timepoint.T4):
            item = followers[point.value]
            if not isinstance(item, Mapping) or set(item) != {"kind", "options"}:
                raise ValueError(f"immediate follower {point.value} must contain exactly kind and options")
            options = item["options"]
            if not isinstance(options, list):
                raise ValueError(f"immediate follower {point.value}.options must be an array")
            parsed_options = []
            for option in options:
                if not isinstance(option, Mapping) or set(option) != {"text", "token_ids"}:
                    raise ValueError(f"immediate follower {point.value} options require text and token_ids")
                if not isinstance(option["token_ids"], list):
                    raise ValueError(f"immediate follower {point.value} token_ids must be an array")
                parsed_options.append(FollowerOption(option["text"], tuple(option["token_ids"])))
            follower_rules.append(ImmediateFollowerRule(point, item["kind"], tuple(parsed_options)))
        return cls(
            value["tokenizer_id"], value["tokenizer_revision"], tuple(definitions), tuple(follower_rules)
        )

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer_id, str) or not self.tokenizer_id:
            raise ValueError("tokenizer_id must be a nonempty string")
        if not isinstance(self.tokenizer_revision, str) or not self.tokenizer_revision:
            raise ValueError("tokenizer_revision must be an immutable nonempty revision")
        if tuple(item.timepoint for item in self.sentinels) != tuple(Timepoint):
            raise ValueError("sentinels must be ordered exactly T0--T4")
        texts = [item.text for item in self.sentinels]
        if len(texts) != len(set(texts)):
            raise ValueError("sentinel texts must be unique")
        if (
            not isinstance(self.immediate_followers, tuple)
            or any(not isinstance(rule, ImmediateFollowerRule) for rule in self.immediate_followers)
        ):
            raise ValueError("immediate_followers must be a tuple of ImmediateFollowerRule records")
        if tuple(rule.timepoint for rule in self.immediate_followers) != (Timepoint.T3, Timepoint.T4):
            raise ValueError("immediate follower rules must be ordered exactly T3, T4")


@dataclass(frozen=True)
class ResolvedBoundary:
    timepoint: Timepoint
    sentinel_text: str
    sentinel_token_ids: tuple[int, ...]
    sentinel_start: int
    sentinel_end: int
    capture_position: int
    immediate_follower_kind: str | None = None
    immediate_follower_text: str | None = None
    immediate_follower_token_ids: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "timepoint": self.timepoint.value,
            "sentinel_text": self.sentinel_text,
            "sentinel_token_ids": list(self.sentinel_token_ids),
            "sentinel_start": self.sentinel_start,
            "sentinel_end": self.sentinel_end,
            "capture_position": self.capture_position,
        }
        if self.immediate_follower_kind is not None:
            value.update(
                {
                    "immediate_follower_kind": self.immediate_follower_kind,
                    "immediate_follower_text": self.immediate_follower_text,
                    "immediate_follower_token_ids": list(self.immediate_follower_token_ids or ()),
                }
            )
        return value


@dataclass(frozen=True)
class ResolvedPrompt:
    prompt: str
    tokenizer_id: str
    tokenizer_revision: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    boundaries: tuple[ResolvedBoundary, ...]
    content_sha256: str

    @property
    def positions(self) -> Mapping[Timepoint, int]:
        return MappingProxyType({item.timepoint: item.capture_position for item in self.boundaries})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "jump.token-boundaries/v2",
            "prompt": self.prompt,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "boundaries": [item.to_dict() for item in self.boundaries],
            "content_sha256": self.content_sha256,
        }

    def verify(self) -> None:
        payload = self.to_dict()
        claimed = payload.pop("content_sha256")
        if _sha256(payload) != claimed:
            raise ValueError("resolved prompt content hash mismatch")


def resolve_boundaries(
    tokenizer: TokenizerLike,
    prompt: str,
    manifest: BoundaryManifest,
) -> ResolvedPrompt:
    """Resolve every preregistered sentinel exactly once in text and token space.

    A sentinel's isolated tokenization must equal the manifest IDs, and that
    exact sequence must occur once in the full tokenization. This rejects
    context-sensitive retokenization rather than guessing a nearby boundary.
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    encoded = tokenizer(prompt, add_special_tokens=True, return_attention_mask=True)
    input_ids = _one_sequence(encoded.get("input_ids"), "input_ids")
    mask = _one_sequence(encoded.get("attention_mask"), "attention_mask")
    if len(input_ids) != len(mask) or not input_ids:
        raise ValueError("tokenizer input_ids and attention_mask must have the same nonzero length")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("attention_mask must contain only 0/1 values")

    boundaries: list[ResolvedBoundary] = []
    follower_rules = {rule.timepoint: rule for rule in manifest.immediate_followers}
    for definition in manifest.sentinels:
        if prompt.count(definition.text) != 1:
            raise ValueError(f"sentinel {definition.timepoint.value} must occur exactly once in prompt text")
        isolated = tokenizer(definition.text, add_special_tokens=False, return_attention_mask=False)
        isolated_ids = tuple(_one_sequence(isolated.get("input_ids"), "sentinel input_ids"))
        if isolated_ids != definition.token_ids:
            raise ValueError(
                f"sentinel {definition.timepoint.value} token IDs drifted: "
                f"expected {definition.token_ids}, got {isolated_ids}"
            )
        matches = _subsequence_positions(tuple(input_ids), definition.token_ids)
        if len(matches) != 1:
            raise ValueError(
                f"sentinel {definition.timepoint.value} must resolve exactly once in full tokenization; "
                "missing matches indicate a retokenized boundary"
            )
        start = matches[0]
        end = start + len(definition.token_ids)
        if any(mask[index] != 1 for index in range(start, end)):
            raise ValueError(f"sentinel {definition.timepoint.value} resolves to a masked token")
        resolved_follower: FollowerOption | None = None
        follower_rule = follower_rules.get(definition.timepoint)
        if follower_rule is not None:
            text_end = prompt.index(definition.text) + len(definition.text)
            for option in follower_rule.options:
                isolated_follower = tokenizer(
                    option.text, add_special_tokens=False, return_attention_mask=False
                )
                isolated_follower_ids = tuple(
                    _one_sequence(isolated_follower.get("input_ids"), "immediate follower input_ids")
                )
                if isolated_follower_ids != option.token_ids:
                    raise ValueError(
                        f"immediate follower {definition.timepoint.value} token IDs drifted: "
                        f"expected {option.token_ids}, got {isolated_follower_ids}"
                    )
                if (
                    prompt.startswith(option.text, text_end)
                    and tuple(input_ids[end : end + len(option.token_ids)]) == option.token_ids
                ):
                    resolved_follower = option
            if resolved_follower is None:
                raise ValueError(
                    f"sentinel {definition.timepoint.value} must end immediately before "
                    f"{follower_rule.kind}"
                )
            follower_end = end + len(resolved_follower.token_ids)
            if any(mask[index] != 1 for index in range(end, follower_end)):
                raise ValueError(f"immediate follower {definition.timepoint.value} resolves to a masked token")
        boundaries.append(
            ResolvedBoundary(
                timepoint=definition.timepoint,
                sentinel_text=definition.text,
                sentinel_token_ids=definition.token_ids,
                sentinel_start=start,
                sentinel_end=end,
                capture_position=end - 1,
                immediate_follower_kind=follower_rule.kind if follower_rule else None,
                immediate_follower_text=resolved_follower.text if resolved_follower else None,
                immediate_follower_token_ids=resolved_follower.token_ids if resolved_follower else None,
            )
        )
    positions = [item.capture_position for item in boundaries]
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise ValueError("T0--T4 token boundaries must be unique and ordered")

    payload = {
        "schema_version": "jump.token-boundaries/v2",
        "prompt": prompt,
        "tokenizer_id": manifest.tokenizer_id,
        "tokenizer_revision": manifest.tokenizer_revision,
        "input_ids": input_ids,
        "attention_mask": mask,
        "boundaries": [item.to_dict() for item in boundaries],
    }
    resolved = ResolvedPrompt(
        prompt=prompt,
        tokenizer_id=manifest.tokenizer_id,
        tokenizer_revision=manifest.tokenizer_revision,
        input_ids=tuple(input_ids),
        attention_mask=tuple(mask),
        boundaries=tuple(boundaries),
        content_sha256=_sha256(payload),
    )
    resolved.verify()
    return resolved


def require_matched_prompt_lengths(*prompts: ResolvedPrompt) -> int:
    if len(prompts) < 2:
        raise ValueError("matched patching requires at least two resolved prompts")
    for prompt in prompts:
        prompt.verify()
    lengths = {len(prompt.input_ids) for prompt in prompts}
    masks = {len(prompt.attention_mask) for prompt in prompts}
    if len(lengths) != 1 or lengths != masks:
        raise ValueError("clean/corrupt/restore prompts must have matched tokenized lengths and masks")
    if len({prompt.attention_mask for prompt in prompts}) != 1:
        raise ValueError("clean/corrupt/restore attention masks must match exactly")
    for point in Timepoint:
        if len({prompt.positions[point] for prompt in prompts}) != 1:
            raise ValueError(f"clean/corrupt/restore {point.value} boundary positions must match")
    return next(iter(lengths))


def _one_sequence(value: Any, name: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise ValueError(f"tokenizer {name} must be one integer sequence")
    return value


def _subsequence_positions(values: tuple[int, ...], needle: tuple[int, ...]) -> list[int]:
    return [index for index in range(len(values) - len(needle) + 1) if values[index : index + len(needle)] == needle]


def _sha256(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()
