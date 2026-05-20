from types import SimpleNamespace

import pytest

from src.config.model_configs import APIProvider, ReasoningParseMode, ToolArgumentParseMode
from src.llm_models.model_client.openai_client import (
    OPENAI_COMPAT_EXTRA_CONTENT_PROVIDER_KEY,
    OPENAI_COMPAT_REASONING_CONTENT_KEY,
    _OpenAIStreamAccumulator,
    _build_reasoning_key,
    _convert_messages,
    _default_normal_response_parser,
    _parse_tool_arguments,
    _sanitize_messages_for_toolless_request,
)
from src.llm_models.payload_content.message import Message, RoleType, TextMessagePart
from src.llm_models.payload_content.tool_option import ToolCall


@pytest.mark.parametrize("parse_mode", list(ToolArgumentParseMode))
def test_parse_tool_arguments_treats_blank_arguments_as_empty_dict(parse_mode: ToolArgumentParseMode) -> None:
    assert _parse_tool_arguments("", parse_mode, None) == {}
    assert _parse_tool_arguments("   ", parse_mode, None) == {}


def test_normal_response_parser_accepts_empty_string_arguments_for_parameterless_tool() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="finish-call",
                            type="function",
                            function=SimpleNamespace(name="finish", arguments=""),
                        )
                    ],
                ),
            )
        ],
        usage=None,
        model="glm-5.1",
    )

    api_response, usage_record = _default_normal_response_parser(
        response,
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
        reasoning_key=None,
    )

    assert len(api_response.tool_calls) == 1
    assert api_response.tool_calls[0].func_name == "finish"
    assert api_response.tool_calls[0].args == {}
    assert usage_record is None


def test_sanitize_messages_for_toolless_request_drops_assistant_tool_call_without_parts() -> None:
    messages = [
        Message(
            role=RoleType.Assistant,
            tool_calls=[
                ToolCall(
                    call_id="call_1",
                    func_name="mute_user",
                    args={"target": "alice"},
                )
            ],
        ),
        Message(
            role=RoleType.User,
            parts=[TextMessagePart(text="继续")],
        ),
    ]

    sanitized_messages = _sanitize_messages_for_toolless_request(messages)

    assert len(sanitized_messages) == 1
    assert sanitized_messages[0].role == RoleType.User


def test_normal_response_parser_ignores_reasoning_field_for_non_openrouter_provider() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="正式回复",
                    reasoning="推理内容",
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
        model="openrouter/test-model",
    )

    api_response, usage_record = _default_normal_response_parser(
        response,
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
        reasoning_key=_build_reasoning_key(
            APIProvider(name="test", base_url="https://openrouter.ai.example.com/api/v1", api_key="test")
        ),
    )

    assert api_response.content == "正式回复"
    assert api_response.reasoning_content is None
    assert usage_record is None


def test_normal_response_parser_reads_provider_reasoning_field_for_reasoning_domains() -> None:
    provider_urls = [
        "https://openrouter.ai/compatible-api",
        "https://api.groq.com/openai/v1",
    ]

    for provider_url in provider_urls:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="正式回复",
                        reasoning="推理内容",
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
            model="test-model",
        )

        api_response, usage_record = _default_normal_response_parser(
            response,
            reasoning_parse_mode=ReasoningParseMode.AUTO,
            tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
            reasoning_key=_build_reasoning_key(
                APIProvider(name="reasoning-provider", base_url=provider_url, api_key="test")
            ),
        )

        assert api_response.content == "正式回复"
        assert api_response.reasoning_content == "推理内容"
        assert usage_record is None


def test_stream_accumulator_reads_openrouter_reasoning_delta_field() -> None:
    accumulator = _OpenAIStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
        reasoning_key=_build_reasoning_key(
            APIProvider(name="openrouter", base_url="https://openrouter.ai/compatible-api", api_key="test")
        ),
    )
    try:
        accumulator.process_delta(SimpleNamespace(reasoning="流式推理", content=None, tool_calls=None))
        accumulator.process_delta(SimpleNamespace(content="正式回复", tool_calls=None))

        api_response = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "正式回复"
    assert api_response.reasoning_content == "流式推理"


def test_normal_response_parser_attaches_reasoning_to_first_tool_call_extra_content() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    reasoning_content="推理内容",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(name="finish", arguments="{}"),
                        )
                    ],
                ),
            )
        ],
        usage=None,
        model="deepseek-reasoner",
    )

    api_response, _ = _default_normal_response_parser(
        response,
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
        reasoning_key="reasoning_content",
    )

    assert api_response.reasoning_content == "推理内容"
    assert api_response.tool_calls is not None
    assert api_response.tool_calls[0].extra_content == {
        OPENAI_COMPAT_EXTRA_CONTENT_PROVIDER_KEY: {
            OPENAI_COMPAT_REASONING_CONTENT_KEY: "推理内容",
        }
    }


def test_convert_messages_roundtrips_provider_reasoning_field_from_tool_call_extra_content() -> None:
    assistant_message = Message(
        role=RoleType.Assistant,
        tool_calls=[
            ToolCall(
                call_id="call_1",
                func_name="finish",
                args={},
                extra_content={
                    OPENAI_COMPAT_EXTRA_CONTENT_PROVIDER_KEY: {
                        OPENAI_COMPAT_REASONING_CONTENT_KEY: "推理内容",
                    }
                },
            )
        ],
    )

    converted_messages = _convert_messages([assistant_message], reasoning_key="reasoning_content")

    assert converted_messages[0]["role"] == "assistant"
    assert converted_messages[0]["content"] is None
    assert converted_messages[0]["reasoning_content"] == "推理内容"
