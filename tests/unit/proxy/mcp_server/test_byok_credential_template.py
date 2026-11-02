"""BYOK servers can compose the outbound credential from ``mcp_info.byok_credential_template``."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest, UserAPIKeyAuth
from litellm.proxy.mcp_server.operations import _get_byok_credential
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _server(mcp_info: dict | None) -> MCPServer:
    return MCPServer(
        server_id="srv-1",
        name="srv",
        url="http://localhost:1/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.basic,
        is_byok=True,
        mcp_info=mcp_info,
    )


async def _resolve(server: MCPServer, stored: str | None, auth: UserAPIKeyAuth) -> str | None:
    with (
        patch("litellm.proxy.mcp_server.operations.get_cached_byok_credential", return_value=None),
        patch("litellm.proxy.mcp_server.operations.cache_byok_credential"),
        patch("litellm.proxy.mcp_server.db.get_user_credential", new=AsyncMock(return_value=stored)),
        patch("litellm.proxy.proxy_server.prisma_client", new=MagicMock()),
    ):
        return await _get_byok_credential(server, auth)


@pytest.mark.asyncio
async def test_template_composes_user_email_and_credential():
    auth = UserAPIKeyAuth(user_id="u-1", user_email="alice@example.com")
    server = _server({"byok_credential_template": "{user_email}:{credential}"})
    assert await _resolve(server, "secret-key", auth) == "alice@example.com:secret-key"


@pytest.mark.asyncio
async def test_template_applies_to_cached_credential():
    auth = UserAPIKeyAuth(user_id="u-1", user_email="alice@example.com")
    server = _server({"byok_credential_template": "{user_id}:{credential}"})
    cached = MagicMock(credential="secret-key")
    with patch("litellm.proxy.mcp_server.operations.get_cached_byok_credential", return_value=cached):
        assert await _get_byok_credential(server, auth) == "u-1:secret-key"


@pytest.mark.asyncio
async def test_without_template_credential_is_forwarded_unchanged():
    auth = UserAPIKeyAuth(user_id="u-1", user_email="alice@example.com")
    assert await _resolve(_server(None), "secret-key", auth) == "secret-key"


@pytest.mark.asyncio
async def test_placeholders_inside_values_are_not_expanded():
    auth = UserAPIKeyAuth(user_id="u-1", user_email="alice@example.com")
    server = _server({"byok_credential_template": "{credential}:{user_id}"})
    assert await _resolve(server, "{user_email}", auth) == "{user_email}:u-1"


@pytest.mark.asyncio
async def test_missing_credential_stays_none():
    auth = UserAPIKeyAuth(user_id="u-1", user_email="alice@example.com")
    server = _server({"byok_credential_template": "{user_email}:{credential}"})
    assert await _resolve(server, None, auth) is None


@pytest.mark.parametrize(
    "template",
    ["{user_email}", "{credential}:{env.SECRET}", 123],
)
def test_invalid_templates_are_rejected_on_create_and_update(template):
    with pytest.raises(ValidationError):
        NewMCPServerRequest(
            server_name="srv", url="http://localhost:1/mcp", mcp_info={"byok_credential_template": template}
        )
    with pytest.raises(ValidationError):
        UpdateMCPServerRequest(server_id="srv-1", mcp_info={"byok_credential_template": template})


def test_valid_template_is_accepted_on_create():
    request = NewMCPServerRequest(
        server_name="srv",
        url="http://localhost:1/mcp",
        mcp_info={"byok_credential_template": "{user_email}:{credential}"},
    )
    assert request.mcp_info == {"byok_credential_template": "{user_email}:{credential}"}


def test_stored_server_with_unvalidated_template_still_loads():
    """Rows saved before validation existed must keep loading; unknown placeholders stay literal."""
    assert _server({"byok_credential_template": "{credential}:{env.SECRET}"}).mcp_info is not None
