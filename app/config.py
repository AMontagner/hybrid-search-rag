from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    rag_mode: Literal['demo', 'cloud'] = 'demo'
    data_dir: str = 'data'
    anthropic_api_key: str = ''
    anthropic_model: str = 'claude-sonnet-4-5'
    cohere_api_key: str = ''
    cohere_embed_model: str = 'embed-english-v3.0'
    cohere_rerank_model: str = 'rerank-v3.5'
    pinecone_api_key: str = ''
    pinecone_host: str = ''
    pinecone_namespace: str = 'internal-docs'
    app_api_key: str = ''
