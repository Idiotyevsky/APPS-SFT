from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MessageSpan:
    message_index: int
    role: str
    trainable: bool
    start: int
    end: int


def _ids(value: Any) -> list[int]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            value = value[0]
        return list(value)
    if hasattr(value, "tolist"):  # torch tensor / numpy array
        return _ids(value.tolist())
    if hasattr(value, "keys") and hasattr(value, "get"):  # BatchEncoding
        return _ids(value["input_ids"])
    if hasattr(value, "ids"):  # tokenizers Encoding
        return list(value.ids)
    return list(value)


def _public_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in message.items() if key != "trainable"} for message in messages]

def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public = _public_messages(messages)
    adapted: list[dict[str, Any]] = []
    for message in public:
        value = dict(message)
        if "tool_calls" in value:
            value["tool_calls"] = [
                {"type": "function", "function": call}
                for call in value["tool_calls"]
            ]
        adapted.append(value)
    return adapted


def tokenize_with_loss_mask(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    public = _template_messages(messages)
    prefixes: list[list[int]] = [[]]
    for end in range(1, len(public) + 1):
        rendered = tokenizer.apply_chat_template(public[:end], tools=tools, tokenize=True, add_generation_prompt=False)
        prefixes.append(_ids(rendered))
    input_ids = prefixes[-1]
    spans: list[MessageSpan] = []
    prior = 0
    for index, message in enumerate(messages):
        current = len(prefixes[index + 1])
        if prefixes[index + 1][:prior] != input_ids[:prior]:
            raise ValueError("chat template is not prefix-stable; cannot prove message loss spans")
        spans.append(MessageSpan(index, message["role"], bool(message.get("trainable")), prior, current))
        prior = current
    labels = [-100] * len(input_ids)
    for span in spans:
        if span.role == "assistant" and span.trainable:
            labels[span.start:span.end] = input_ids[span.start:span.end]
    attention_mask = [1] * len(input_ids)
    validate_loss_mask(messages, input_ids, attention_mask, labels, spans)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "spans": [span.__dict__ if hasattr(span, "__dict__") else {"message_index": span.message_index, "role": span.role, "trainable": span.trainable, "start": span.start, "end": span.end} for span in spans]}


def validate_loss_mask(messages, input_ids, attention_mask, labels, spans) -> None:
    if not (len(input_ids) == len(attention_mask) == len(labels)) or not any(value != -100 for value in labels):
        raise ValueError("invalid token arrays or no supervised token")
    for span in spans:
        supervised = labels[span.start:span.end]
        attention = attention_mask[span.start:span.end]
        should_train = span.role == "assistant" and span.trainable
        if should_train and (not supervised or any(value == -100 for value in supervised)):
            raise ValueError("trainable assistant span is not fully supervised")
        if not should_train and any(value != -100 for value in supervised):
            raise ValueError("non-trainable span contains labels")
        if any(value != 1 for value in attention):
            raise ValueError("non-padding context must remain visible")


def decode_supervised(tokenizer, input_ids: list[int], labels: list[int]) -> str:
    return tokenizer.decode([token for token, label in zip(input_ids, labels) if label != -100], skip_special_tokens=False)

