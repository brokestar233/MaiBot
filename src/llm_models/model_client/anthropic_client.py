from typing import Any, Dict, List, Tuple, cast
from uuid import uuid4

import asyncio

from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic

from src.common.logger import get_logger
from src.config.model_configs import APIProvider
from src.llm_models.exceptions import (
    EmptyResponseException,
    NetworkConnectionError,
    ReqAbortException,
    RespNotOkException,
    RespParseException,
)
from src.llm_models.openai_compat import normalize_openai_base_url, split_openai_request_overrides
from src.llm_models.payload_content.message import ImageMessagePart, Message, RoleType, TextMessagePart
from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType
from src.llm_models.payload_content.tool_option import ToolCall, ToolOption

from .adapter_base import AdapterClient, ProviderResponseParser, ProviderStreamResponseHandler, await_task_with_interrupt
from .base_client import (
    APIResponse,
    AudioTranscriptionRequest,
    EmbeddingRequest,
    ResponseRequest,
    UsageTuple,
    client_registry,
)
from ..request_snapshot import (
    attach_request_snapshot,
    has_request_snapshot,
    save_failed_request_snapshot,
    serialize_audio_request_snapshot,
    serialize_embedding_request_snapshot,
    serialize_response_request_snapshot,
)

logger = get_logger("llm_models")

ANTHROPIC_EXTRA_CONTENT_PROVIDER_KEY = "anthropic"
ANTHROPIC_EXTRA_CONTENT_THINKING_BLOCKS_KEY = "thinking_blocks"
"""Anthropic 工具调用附加信息中的思考块缓存字段。"""

ANTHROPIC_MESSAGES_RESERVED_EXTRA_BODY_KEYS = {
    "container",
    "inference_geo",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "output_config",
    "service_tier",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
}
"""Anthropic Messages API 由 SDK 显式承载的参数集合。"""

SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES: dict[str, str] = {
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
"""Anthropic Messages API 支持的图片 MIME 类型映射。"""


def _build_fallback_tool_call_id(prefix: str) -> str:
    normalized_prefix = str(prefix).strip() or "tool_use"
    return f"{normalized_prefix}_{uuid4().hex}"


def _build_empty_object_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
    }


def _build_text_content(message: Message) -> str:
    return "".join(part.text for part in message.parts if isinstance(part, TextMessagePart))


def _convert_response_format(response_format: RespFormat | None) -> Dict[str, Any] | None:
    if response_format is None:
        return None

    if response_format.format_type == RespFormatType.TEXT:
        return None

    if response_format.format_type == RespFormatType.JSON_OBJ:
        return {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                },
            }
        }

    schema_payload = response_format.get_schema_object()
    if not isinstance(schema_payload, dict):
        raise ValueError("Anthropic 结构化输出需要对象级 JSON Schema")

    return {
        "format": {
            "type": "json_schema",
            "schema": schema_payload,
        }
    }


def _convert_tool_options(tool_options: List[ToolOption]) -> List[Dict[str, Any]]:
    converted_tools: List[Dict[str, Any]] = []
    for tool_option in tool_options:
        converted_tools.append(
            {
                "name": tool_option.name,
                "description": tool_option.description,
                "input_schema": tool_option.parameters_schema or _build_empty_object_schema(),
            }
        )
    return converted_tools


def _sanitize_messages_for_toolless_request(messages: List[Message]) -> List[Message]:
    """在无工具请求时清洗历史工具调用链，避免 Anthropic 拒收消息。"""
    sanitized_messages: List[Message] = []

    for message in messages:
        if message.role == RoleType.Tool:
            continue

        if message.role == RoleType.Assistant and message.tool_calls:
            if not message.parts:
                continue
            sanitized_messages.append(
                Message(
                    role=message.role,
                    parts=list(message.parts),
                    tool_call_id=message.tool_call_id,
                    tool_name=message.tool_name,
                    tool_calls=None,
                )
            )
            continue

        sanitized_messages.append(message)

    return sanitized_messages


def _convert_part_to_content_block(part: TextMessagePart | ImageMessagePart, *, allow_image: bool) -> Dict[str, Any]:
    if isinstance(part, TextMessagePart):
        return {
            "type": "text",
            "text": part.text,
        }

    if not allow_image:
        raise RespParseException(None, "Anthropic 历史消息仅支持在 user 消息中携带图片。")

    media_type = SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES.get(part.normalized_image_format)
    if media_type is None:
        raise RespParseException(None, f"Anthropic 不支持图片格式: {part.image_format}")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": part.image_base64,
        },
    }


def _convert_messages(messages: List[Message]) -> Tuple[str | None, List[Dict[str, Any]]]:
    system_blocks: List[str] = []
    converted_messages: List[Dict[str, Any]] = []

    for message in messages:
        if message.role == RoleType.System:
            system_text = _build_text_content(message).strip()
            if system_text:
                system_blocks.append(system_text)
            continue

        if message.role == RoleType.User:
            converted_messages.append(
                {
                    "role": "user",
                    "content": [
                        _convert_part_to_content_block(part, allow_image=True)
                        for part in message.parts
                    ],
                }
            )
            continue

        if message.role == RoleType.Assistant:
            non_tool_content_blocks = [
                _convert_part_to_content_block(part, allow_image=False)
                for part in message.parts
            ]
            content_blocks: List[Dict[str, Any]] = []
            tool_calls = message.tool_calls or []
            for tool_call_index, tool_call in enumerate(tool_calls):
                content_blocks.extend(_extract_anthropic_thinking_blocks(tool_call))
                if tool_call_index == 0:
                    content_blocks.extend(non_tool_content_blocks)
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.call_id or _build_fallback_tool_call_id(tool_call.func_name),
                        "name": tool_call.func_name,
                        "input": tool_call.args or {},
                    }
                )
            if not tool_calls:
                content_blocks.extend(non_tool_content_blocks)
            converted_messages.append(
                {
                    "role": "assistant",
                    "content": content_blocks,
                }
            )
            continue

        if message.role == RoleType.Tool:
            if not message.tool_call_id:
                raise RespParseException(None, "Tool 消息缺少 tool_call_id")
            tool_result_text = _build_text_content(message)
            converted_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": tool_result_text,
                        }
                    ],
                }
            )
            continue

        raise RespParseException(None, f"不支持的消息角色: {message.role}")

    system_prompt = "\n\n".join(block for block in system_blocks if block).strip() or None
    return system_prompt, converted_messages


def _build_anthropic_thinking_block(block: Any) -> Dict[str, Any] | None:
    """将 Anthropic thinking/redacted_thinking block 规范化为可回传结构。"""

    block_type = getattr(block, "type", None)
    if block_type == "thinking":
        signature = getattr(block, "signature", None)
        thinking = getattr(block, "thinking", None)
        payload: Dict[str, Any] = {
            "type": "thinking",
            "thinking": thinking if isinstance(thinking, str) else "",
        }
        if isinstance(signature, str) and signature:
            payload["signature"] = signature
        return payload

    if block_type == "redacted_thinking":
        data = getattr(block, "data", None)
        if isinstance(data, str) and data:
            return {
                "type": "redacted_thinking",
                "data": data,
            }

    return None


def _build_anthropic_tool_call_extra_content(thinking_blocks: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """将 Anthropic 思考块列表编码到工具调用附加信息中。"""

    if not thinking_blocks:
        return None

    normalized_blocks = [dict(thinking_block) for thinking_block in thinking_blocks if isinstance(thinking_block, dict)]
    if not normalized_blocks:
        return None

    return {
        ANTHROPIC_EXTRA_CONTENT_PROVIDER_KEY: {
            ANTHROPIC_EXTRA_CONTENT_THINKING_BLOCKS_KEY: normalized_blocks,
        }
    }


def _extract_anthropic_thinking_blocks(tool_call: ToolCall) -> List[Dict[str, Any]]:
    """从工具调用附加信息中提取 Anthropic thinking block。"""

    if not tool_call.extra_content:
        return []

    provider_payload = tool_call.extra_content.get(ANTHROPIC_EXTRA_CONTENT_PROVIDER_KEY)
    if not isinstance(provider_payload, dict):
        return []

    raw_thinking_blocks = provider_payload.get(ANTHROPIC_EXTRA_CONTENT_THINKING_BLOCKS_KEY)
    if not isinstance(raw_thinking_blocks, list):
        return []

    normalized_blocks: List[Dict[str, Any]] = []
    for raw_thinking_block in raw_thinking_blocks:
        if not isinstance(raw_thinking_block, dict):
            continue

        block_type = str(raw_thinking_block.get("type") or "").strip()
        if block_type == "thinking":
            normalized_block: Dict[str, Any] = {
                "type": "thinking",
                "thinking": str(raw_thinking_block.get("thinking") or ""),
            }
            signature = raw_thinking_block.get("signature")
            if isinstance(signature, str) and signature:
                normalized_block["signature"] = signature
            normalized_blocks.append(normalized_block)
            continue

        if block_type == "redacted_thinking":
            data = raw_thinking_block.get("data")
            if isinstance(data, str) and data:
                normalized_blocks.append(
                    {
                        "type": "redacted_thinking",
                        "data": data,
                    }
                )

    return normalized_blocks


def _extract_usage_record(raw_usage: Any) -> UsageTuple | None:
    if raw_usage is None:
        return None

    input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
    cache_creation_input_tokens = int(getattr(raw_usage, "cache_creation_input_tokens", 0) or 0)
    cache_read_input_tokens = int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0)
    prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
    prompt_cache_hit_tokens = cache_read_input_tokens
    prompt_cache_miss_tokens = max(0, prompt_tokens - prompt_cache_hit_tokens)
    return (
        prompt_tokens,
        output_tokens,
        prompt_tokens + output_tokens,
        prompt_cache_hit_tokens,
        prompt_cache_miss_tokens,
    )


def _default_response_parser(resp: Any) -> Tuple[APIResponse, UsageTuple | None]:
    response = APIResponse()
    text_blocks: List[str] = []
    reasoning_blocks: List[str] = []
    tool_calls: List[ToolCall] = []
    pending_thinking_blocks: List[Dict[str, Any]] = []

    for block in getattr(resp, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                text_blocks.append(text)
            continue

        if block_type in {"thinking", "redacted_thinking"}:
            thinking_block = _build_anthropic_thinking_block(block)
            if thinking_block is not None:
                pending_thinking_blocks.append(thinking_block)

        if block_type == "thinking":
            thinking = getattr(block, "thinking", None)
            if isinstance(thinking, str) and thinking:
                reasoning_blocks.append(thinking)
            continue

        if block_type in {"tool_use", "server_tool_use"}:
            extra_content = _build_anthropic_tool_call_extra_content(pending_thinking_blocks)
            tool_calls.append(
                ToolCall(
                    call_id=str(getattr(block, "id", None) or _build_fallback_tool_call_id("tool_use")),
                    func_name=str(getattr(block, "name", "") or ""),
                    args=cast(Dict[str, Any] | None, getattr(block, "input", None)),
                    extra_content=extra_content,
                )
            )
            pending_thinking_blocks = []

    response.content = "".join(text_blocks).strip() or None
    response.reasoning_content = "\n\n".join(reasoning_blocks).strip() or None
    response.tool_calls = tool_calls or None
    response.raw_data = resp

    if not response.content and not response.tool_calls:
        raise EmptyResponseException(resp)

    return response, _extract_usage_record(getattr(resp, "usage", None))


@client_registry.register_client_class("anthropic")
class AnthropicClient(AdapterClient[Any, Any]):
    """Anthropic Messages API 原生客户端。"""

    client: AsyncAnthropic

    def __init__(self, api_provider: APIProvider) -> None:
        super().__init__(api_provider)
        default_headers = dict(api_provider.default_headers)
        default_headers.setdefault("anthropic-version", "2023-06-01")
        self.client = AsyncAnthropic(
            api_key=api_provider.api_key or None,
            base_url=normalize_openai_base_url(api_provider.base_url),
            timeout=api_provider.timeout,
            max_retries=api_provider.max_retry,
            default_headers=default_headers or None,
            default_query=dict(api_provider.default_query) or None,
        )

    def _build_default_stream_response_handler(
        self,
        request: ResponseRequest,
    ) -> ProviderStreamResponseHandler[Any]:
        del request

        async def default_stream_handler(
            resp_stream: Any,
            flag: asyncio.Event | None,
        ) -> Tuple[APIResponse, UsageTuple | None]:
            del flag
            return _default_response_parser(resp_stream)

        return default_stream_handler

    def _build_default_response_parser(
        self,
        request: ResponseRequest,
    ) -> ProviderResponseParser[Any]:
        del request
        return _default_response_parser

    async def _execute_response_request(
        self,
        request: ResponseRequest,
        stream_response_handler: ProviderStreamResponseHandler[Any],
        response_parser: ProviderResponseParser[Any],
    ) -> Tuple[APIResponse, UsageTuple | None]:
        del stream_response_handler

        if request.stream_response_handler is not None or request.async_response_parser is not None:
            raise RespParseException(None, "Anthropic 客户端暂不支持自定义流式处理器或响应解析器")

        snapshot_provider_request = {
            "base_url": self.api_provider.base_url,
            "endpoint": "/messages",
            "method": "POST",
            "operation": "messages.create",
            "request_kwargs": {},
        }
        model_info = request.model_info

        try:
            request_messages = (
                list(request.message_list)
                if request.tool_options
                else _sanitize_messages_for_toolless_request(request.message_list)
            )
            system_prompt, messages_payload = _convert_messages(request_messages)
            tools_payload = _convert_tool_options(request.tool_options) if request.tool_options else None
            output_config = _convert_response_format(request.response_format)
            request_overrides = split_openai_request_overrides(
                request.extra_params,
                reserved_body_keys=ANTHROPIC_MESSAGES_RESERVED_EXTRA_BODY_KEYS,
            )

            request_kwargs: Dict[str, Any] = {
                "max_tokens": max(1, int(request.max_tokens or 1)),
                "messages": messages_payload,
                "model": model_info.model_identifier,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt
            if tools_payload:
                request_kwargs["tools"] = tools_payload
            if output_config is not None:
                request_kwargs["output_config"] = output_config
            if request.temperature is not None:
                request_kwargs["temperature"] = request.temperature

            for key in [
                "container",
                "inference_geo",
                "metadata",
                "service_tier",
                "stop_sequences",
                "thinking",
                "tool_choice",
                "top_k",
                "top_p",
            ]:
                if key in request_overrides.extra_body:
                    request_kwargs[key] = request_overrides.extra_body.pop(key)

            snapshot_provider_request["request_kwargs"] = {
                **request_kwargs,
                "extra_body": request_overrides.extra_body or None,
                "extra_headers": request_overrides.extra_headers or None,
                "extra_query": request_overrides.extra_query or None,
                "stream": bool(model_info.force_stream_mode),
            }

            if model_info.force_stream_mode:
                async with self.client.messages.stream(
                    **request_kwargs,
                    extra_body=request_overrides.extra_body or None,
                    extra_headers=request_overrides.extra_headers or None,
                    extra_query=request_overrides.extra_query or None,
                ) as stream:
                    async for _ in stream:
                        if request.interrupt_flag and request.interrupt_flag.is_set():
                            raise ReqAbortException("请求被外部信号中断")
                    raw_response = await stream.get_final_message()
                return response_parser(raw_response)

            response_task: asyncio.Task[Any] = asyncio.create_task(
                self.client.messages.create(
                    **request_kwargs,
                    extra_body=request_overrides.extra_body or None,
                    extra_headers=request_overrides.extra_headers or None,
                    extra_query=request_overrides.extra_query or None,
                )
            )
            raw_response = await await_task_with_interrupt(response_task, request.interrupt_flag)
            return response_parser(raw_response)
        except APIStatusError as exc:
            status_code = int(getattr(exc, "status_code", 500) or 500)
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="anthropic",
                error=exc,
                internal_request=serialize_response_request_snapshot(request),
                model_info=model_info,
                operation="messages.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = RespNotOkException(status_code, str(exc))
            attach_request_snapshot(wrapped_error, snapshot_path)
            raise wrapped_error from exc
        except APIConnectionError as exc:
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="anthropic",
                error=exc,
                internal_request=serialize_response_request_snapshot(request),
                model_info=model_info,
                operation="messages.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = NetworkConnectionError(str(exc))
            attach_request_snapshot(wrapped_error, snapshot_path)
            raise wrapped_error from exc
        except Exception as exc:
            if has_request_snapshot(exc):
                raise
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="anthropic",
                error=exc,
                internal_request=serialize_response_request_snapshot(request),
                model_info=model_info,
                operation="messages.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = (
                exc
                if isinstance(exc, (EmptyResponseException, ReqAbortException, RespParseException))
                else NetworkConnectionError(str(exc))
            )
            attach_request_snapshot(wrapped_error, snapshot_path)
            if wrapped_error is exc:
                raise
            raise wrapped_error from exc

    async def _execute_embedding_request(
        self,
        request: EmbeddingRequest,
    ) -> Tuple[APIResponse, UsageTuple | None]:
        exc = RespNotOkException(400, "Anthropic Messages API 暂不支持 embeddings 接口")
        snapshot_path = save_failed_request_snapshot(
            api_provider=self.api_provider,
            client_type="anthropic",
            error=exc,
            internal_request=serialize_embedding_request_snapshot(request),
            model_info=request.model_info,
            operation="embeddings.unsupported",
            provider_request={
                "base_url": self.api_provider.base_url,
                "endpoint": None,
                "method": None,
                "operation": "embeddings.unsupported",
            },
        )
        attach_request_snapshot(exc, snapshot_path)
        raise exc

    async def _execute_audio_transcription_request(
        self,
        request: AudioTranscriptionRequest,
    ) -> Tuple[APIResponse, UsageTuple | None]:
        exc = RespNotOkException(400, "Anthropic Messages API 暂不支持音频转录接口")
        snapshot_path = save_failed_request_snapshot(
            api_provider=self.api_provider,
            client_type="anthropic",
            error=exc,
            internal_request=serialize_audio_request_snapshot(request),
            model_info=request.model_info,
            operation="audio_transcriptions.unsupported",
            provider_request={
                "base_url": self.api_provider.base_url,
                "endpoint": None,
                "method": None,
                "operation": "audio_transcriptions.unsupported",
            },
        )
        attach_request_snapshot(exc, snapshot_path)
        raise exc

    def get_support_image_formats(self) -> List[str]:
        return list(SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES.keys())
