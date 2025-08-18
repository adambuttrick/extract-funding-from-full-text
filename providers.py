import os
import re
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, List

import requests


class ModelProvider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
    LOCAL_OPENAI = "local_openai"


@dataclass
class ProviderConfig:
    provider: ModelProvider
    default_model: str
    requires_api_key: bool
    default_url: Optional[str] = None


def get_provider_config(provider: ModelProvider) -> ProviderConfig:
    configs = {
        ModelProvider.GEMINI: ProviderConfig(
            provider=ModelProvider.GEMINI,
            default_model="gemini-2.5-flash-lite",
            requires_api_key=True,
            default_url=None,
        ),
        ModelProvider.OLLAMA: ProviderConfig(
            provider=ModelProvider.OLLAMA,
            default_model=None,
            requires_api_key=False,
            default_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        ),
        ModelProvider.OPENAI: ProviderConfig(
            provider=ModelProvider.OPENAI,
            default_model="gpt-4o-mini",
            requires_api_key=True,
            default_url=None,
        ),
        ModelProvider.LOCAL_OPENAI: ProviderConfig(
            provider=ModelProvider.LOCAL_OPENAI,
            default_model=None,
            requires_api_key=False,
            default_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000"),
        ),
    }
    return configs[provider]


def get_available_ollama_models(model_url: Optional[str] = None) -> List[str]:
    url = model_url
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Cannot connect to Ollama at {url}: {e}") from e


def validate_gemini_model(model_id: str) -> bool:
    valid_patterns = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro"
    ]
    
    for pattern in valid_patterns:
        if model_id.startswith(pattern):
            return True
    
    pattern = r"^gemini-\d+\.\d+-[a-z]+(-[a-z]+)*$"
    return bool(re.match(pattern, model_id.lower()))


def validate_openai_model(model_id: str) -> bool:
    """Validate that an OpenAI model ID is in the expected format."""
    valid_patterns = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
        "o3-mini",
    ]
    
    for pattern in valid_patterns:
        if model_id == pattern or model_id.startswith(f"{pattern}-"):
            return True
    
    return False


def validate_ollama_model(model_id: str, model_url: Optional[str] = None) -> None:
    try:
        available_models = get_available_ollama_models(model_url)
        
        model_base = model_id.split(":")[0]
        
        model_found = any(
            model == model_id or model.startswith(f"{model_base}:")
            for model in available_models
        )
        
        if not model_found:
            available_str = (
                ", ".join(sorted(available_models)) if available_models else "none"
            )
            raise ValueError(
                f"Model '{model_id}' is not available in Ollama. "
                f"Available models: {available_str}. "
                f"Install it with: ollama pull {model_id}"
            )
    except ConnectionError:
        raise


def validate_provider_requirements(
    provider: ModelProvider,
    api_key: Optional[str],
    model_url: Optional[str],
    model_id: Optional[str] = None,
    skip_model_validation: bool = False,
) -> None:
    config = get_provider_config(provider)
    
    if config.requires_api_key and not api_key:
        env_var_name = (
            "OPENAI_API_KEY" if provider == ModelProvider.OPENAI else "GEMINI_API_KEY"
        )
        raise ValueError(
            f"{provider.value} requires an API key. "
            f"Set {env_var_name} environment variable or use --api-key"
        )
    
    if model_id and not skip_model_validation:
        if provider == ModelProvider.OLLAMA:
            validate_ollama_model(model_id, model_url)
        elif provider == ModelProvider.GEMINI:
            if not validate_gemini_model(model_id):
                raise ValueError(
                    f"Model '{model_id}' does not appear to be a valid Gemini model. "
                    f"Expected format: gemini-X.X-[flash|pro|lite] "
                    f"(e.g., gemini-2.5-flash, gemini-2.5-pro)"
                )
        elif provider == ModelProvider.OPENAI:
            is_custom_endpoint = model_url and not model_url.startswith(
                "https://api.openai.com"
            )
            if not is_custom_endpoint and not validate_openai_model(model_id):
                raise ValueError(
                    f"Model '{model_id}' does not appear to be a valid OpenAI model. "
                    f"Expected format: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc."
                )