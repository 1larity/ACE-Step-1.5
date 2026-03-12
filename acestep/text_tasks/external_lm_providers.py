"""Provider profiles for external LM integration in ACE-Step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalProviderProfile:
    """Static configuration for an external LM provider."""

    provider_id: str
    label: str
    protocol: str
    default_model: str
    default_base_url: str
    api_key_env: str
    api_key_required: bool
    secret_path_env: str
    secret_file_name: str


_EXTERNAL_PROVIDER_PROFILES: dict[str, ExternalProviderProfile] = {
    "zai": ExternalProviderProfile(
        provider_id="zai",
        label="Z.ai",
        protocol="openai_chat",
        default_model="glm-4.5-flash",
        default_base_url="https://api.z.ai/api/paas/v4/chat/completions",
        api_key_env="ACESTEP_ZAI_API_KEY",
        api_key_required=True,
        secret_path_env="ACESTEP_ZAI_SECRET_PATH",
        secret_file_name="zai_api_key.enc",
    ),
    "openai": ExternalProviderProfile(
        provider_id="openai",
        label="OpenAI",
        protocol="openai_chat",
        default_model="gpt-4o-mini",
        default_base_url="https://api.openai.com/v1/chat/completions",
        api_key_env="ACESTEP_OPENAI_API_KEY",
        api_key_required=True,
        secret_path_env="ACESTEP_OPENAI_SECRET_PATH",
        secret_file_name="openai_api_key.enc",
    ),
    "claude": ExternalProviderProfile(
        provider_id="claude",
        label="Anthropic Claude",
        protocol="anthropic_messages",
        default_model="claude-3-7-sonnet-latest",
        default_base_url="https://api.anthropic.com/v1/messages",
        api_key_env="ACESTEP_ANTHROPIC_API_KEY",
        api_key_required=True,
        secret_path_env="ACESTEP_ANTHROPIC_SECRET_PATH",
        secret_file_name="anthropic_api_key.enc",
    ),
    "ollama": ExternalProviderProfile(
        provider_id="ollama",
        label="Ollama",
        protocol="openai_chat",
        default_model="llama3.1:8b-instruct",
        default_base_url="http://127.0.0.1:11434/v1/chat/completions",
        api_key_env="ACESTEP_OLLAMA_API_KEY",
        api_key_required=False,
        secret_path_env="ACESTEP_OLLAMA_SECRET_PATH",
        secret_file_name="ollama_api_key.enc",
    ),
}


def get_external_provider_profiles() -> dict[str, ExternalProviderProfile]:
    """Return immutable-style copy of provider profile mapping."""
    return dict(_EXTERNAL_PROVIDER_PROFILES)


def get_external_provider_profile(provider: str | None) -> ExternalProviderProfile:
    """Return provider profile, defaulting invalid values to Z.ai profile."""
    if provider and provider in _EXTERNAL_PROVIDER_PROFILES:
        return _EXTERNAL_PROVIDER_PROFILES[provider]
    return _EXTERNAL_PROVIDER_PROFILES["zai"]


def get_external_provider_choices() -> list[tuple[str, str]]:
    """Return ordered provider dropdown choices as ``(label, value)`` tuples."""
    order = ["zai", "openai", "claude", "ollama"]
    return [
        (_EXTERNAL_PROVIDER_PROFILES[provider_id].label, provider_id)
        for provider_id in order
    ]


def build_external_model_choice(provider: str, model: str) -> str:
    """Build LM dropdown token for an external provider/model selection."""
    normalized_provider = provider.strip().lower() if provider else "zai"
    normalized_model = model.strip() if model else get_external_provider_profile(provider).default_model
    return f"external:{normalized_provider}:{normalized_model}"
