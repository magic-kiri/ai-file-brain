from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AiFileBrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AFB_",
        toml_file="settings.toml",
        extra="ignore",
        case_sensitive=False,
    )

    watch_folder: str = r"C:\Users\ASUS\Documents\AIFileBrainTest"
    ollama_url: str = "http://127.0.0.1:11434"
    chroma_path: str = "./chroma-data"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    chunk_size: int = 2000
    chunk_overlap: int = 400
    top_k: int = 5

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        from pydantic_settings import TomlConfigSettingsSource

        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def chroma_path_resolved(self) -> Path:
        return Path(self.chroma_path).expanduser().resolve()
