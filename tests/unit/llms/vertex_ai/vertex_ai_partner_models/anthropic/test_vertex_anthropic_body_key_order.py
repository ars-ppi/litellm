from typing import Final

from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.messages.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import VertexAIAnthropicConfig
from litellm.types.router import GenericLiteLLMParams

_TOOL: Final = {"name": "get_weather", "description": "Weather", "input_schema": {"type": "object", "properties": {}}}


def test_vertex_anthropic_chat_body_puts_messages_last() -> None:
    data: Final = VertexAIAnthropicConfig().transform_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 16, "tools": [_TOOL]},
        litellm_params={},
        headers={},
    )

    assert "system" in data and "tools" in data
    assert list(data)[-1] == "messages"


def test_vertex_anthropic_messages_body_puts_messages_last() -> None:
    data: Final = VertexAIPartnerModelsAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={"max_tokens": 16, "system": "be brief", "tools": [_TOOL]},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "system" in data and "tools" in data
    assert list(data)[-1] == "messages"
