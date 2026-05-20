from types import SimpleNamespace

from src.llm_models.model_client.anthropic_client import (
    ANTHROPIC_EXTRA_CONTENT_PROVIDER_KEY,
    ANTHROPIC_EXTRA_CONTENT_THINKING_BLOCKS_KEY,
    _convert_messages,
    _default_response_parser,
)
from src.llm_models.payload_content.message import MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall


def test_default_response_parser_preserves_thinking_signature_on_tool_call() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="先思考一下", signature="signature-1"),
            SimpleNamespace(type="tool_use", id="toolu_1", name="reply", input={"msg_id": "42"}),
        ],
        usage=None,
    )

    api_response, usage_record = _default_response_parser(response)

    assert api_response.reasoning_content == "先思考一下"
    assert api_response.tool_calls is not None
    assert api_response.tool_calls[0].extra_content == {
        ANTHROPIC_EXTRA_CONTENT_PROVIDER_KEY: {
            ANTHROPIC_EXTRA_CONTENT_THINKING_BLOCKS_KEY: [
                {
                    "type": "thinking",
                    "thinking": "先思考一下",
                    "signature": "signature-1",
                }
            ]
        }
    }
    assert usage_record is None


def test_convert_messages_roundtrips_thinking_signature_before_tool_use() -> None:
    assistant_message = (
        MessageBuilder()
        .set_role(RoleType.Assistant)
        .set_tool_calls(
            [
                ToolCall(
                    call_id="toolu_1",
                    func_name="reply",
                    args={"msg_id": "42"},
                    extra_content={
                        ANTHROPIC_EXTRA_CONTENT_PROVIDER_KEY: {
                            ANTHROPIC_EXTRA_CONTENT_THINKING_BLOCKS_KEY: [
                                {
                                    "type": "thinking",
                                    "thinking": "先思考一下",
                                    "signature": "signature-1",
                                }
                            ]
                        }
                    },
                )
            ]
        )
        .build()
    )

    system_prompt, messages_payload = _convert_messages([assistant_message])

    assert system_prompt is None
    assert messages_payload[0]["role"] == "assistant"
    assert messages_payload[0]["content"] == [
        {
            "type": "thinking",
            "thinking": "先思考一下",
            "signature": "signature-1",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "reply",
            "input": {"msg_id": "42"},
        },
    ]
