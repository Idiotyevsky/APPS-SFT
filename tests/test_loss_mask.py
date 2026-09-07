import json

from synthesis.loss_mask import (
    decode_supervised, tokenize_with_loss_mask,
)
from synthesis.tools import TOOL_SCHEMAS


class PrefixTokenizer:
    def apply_chat_template(
        self, messages, tools, tokenize=True,
        add_generation_prompt=False,
    ):
        text = "<tools>" + json.dumps(tools, sort_keys=True)
        for message in messages:
            text += (
                "<" + message["role"] + ">" +
                json.dumps(message, sort_keys=True)
            )
        return list(text.encode())

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode()


def test_tool_observation_masked_but_visible_and_actions_supervised():
    messages = [
        {"role": "system", "content": "rules", "trainable": False},
        {"role": "user", "content": "buggy code", "trainable": False},
        {
            "role": "assistant",
            "tool_calls": [{
                "name": "run_candidate",
                "arguments": {"input": "1\n"},
            }],
            "trainable": True,
        },
        {
            "role": "tool", "name": "run_candidate",
            "content": '{"status":"ok","stdout":"bad"}',
            "trainable": False,
        },
        {
            "role": "assistant",
            "tool_calls": [{
                "name": "submit",
                "arguments": {"code": "print(1)"},
            }],
            "trainable": True,
        },
    ]
    tokenizer = PrefixTokenizer()
    value = tokenize_with_loss_mask(
        tokenizer, messages, TOOL_SCHEMAS,
    )
    supervised = decode_supervised(
        tokenizer, value["input_ids"], value["labels"],
    )
    assert "run_candidate" in supervised
    assert "submit" in supervised
    assert "stdout" not in supervised
    tool_span = value["spans"][3]
    assert all(
        label == -100
        for label in value["labels"][tool_span["start"]:tool_span["end"]]
    )
    assert all(
        mask == 1
        for mask in value["attention_mask"][
            tool_span["start"]:tool_span["end"]
        ]
    )


def test_failed_assistant_submit_is_fully_masked():
    tokenizer = PrefixTokenizer()
    messages = [
        {"role": "user", "content": "state", "trainable": False},
        {
            "role": "assistant",
            "tool_calls": [{
                "name": "submit",
                "arguments": {"code": "bad()"},
            }],
            "trainable": False,
        },
        {
            "role": "tool", "name": "submit",
            "content": '{"status":"wrong_answer"}',
            "trainable": False,
        },
        {
            "role": "assistant",
            "tool_calls": [{
                "name": "submit",
                "arguments": {"code": "print(1)"},
            }],
            "trainable": True,
        },
    ]
    value = tokenize_with_loss_mask(
        tokenizer, messages, TOOL_SCHEMAS,
    )
    failed = value["spans"][1]
    assert all(
        label == -100
        for label in value["labels"][failed["start"]:failed["end"]]
    )
