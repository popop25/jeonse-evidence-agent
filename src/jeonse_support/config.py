"""Runtime configuration that names, but never exposes, Azure credentials."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

AZURE_OPENAI_ENV_NAMES = (
    "AOAI_API_KEY",
    "AOAI_ENDPOINT",
    "AOAI_DEPLOY_GPT41_MINI",
    "AOAI_MODEL_GPT41_MINI",
    "AOAI_DEPLOY_EMBED_3_SMALL",
    "AOAI_MODEL_EMBED_3_SMALL",
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Only the approved Azure OpenAI variable *names* are recognized."""

    data_mode: Literal["snapshot", "api", "auto"] = "snapshot"
    model_profile: str = "offline-snapshot-demo"
    session_limit: int = 5
    session_intent_bytes: int = 4096

    @property
    def is_offline_demo(self) -> bool:
        return self.data_mode == "snapshot"

    @property
    def azure_configured(self) -> bool:
        return all(bool(os.getenv(name)) for name in AZURE_OPENAI_ENV_NAMES)
    @property
    def azure_configuration_state(self) -> Literal["absent", "partial", "complete"]:
        configured = sum(bool(os.getenv(name)) for name in AZURE_OPENAI_ENV_NAMES)
        if configured == 0:
            return "absent"
        if configured == len(AZURE_OPENAI_ENV_NAMES):
            return "complete"
        return "partial"


    @classmethod
    def from_environment(cls) -> "Settings":
        mode = os.getenv("JEONSE_DATA_MODE", "snapshot").lower()
        if mode not in {"snapshot", "api", "auto"}:
            raise ValueError("INVALID_DATA_MODE")
        azure_state = cls().azure_configuration_state
        if azure_state == "partial":
            raise ValueError("PARTIAL_AZURE_CONFIGURATION")
        profile = "azure-openai" if azure_state == "complete" else "offline-snapshot-demo"
        return cls(data_mode=mode, model_profile=profile)

    def public_metadata(self) -> dict[str, object]:
        return {
            "data_mode": self.data_mode,
            "offline_demo": self.is_offline_demo,
            "model_profile": self.model_profile,
            "provider_configured": self.azure_configured,
        }
