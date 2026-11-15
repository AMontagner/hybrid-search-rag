import secrets
from pathlib import Path
from threading import RLock

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import Settings
from .rag import Engine


def create_app(settings: Settings | None = None):
    settings = settings or Settings()
    engine = Engine(settings)
    lock = RLock()
    api = FastAPI(title='Hybrid Search RAG', version='0.1.0')

    def authorize(x_api_key: str = Header(default='')):
        if settings.app_api_key and not secrets.compare_digest(x_api_key, settings.app_api_key):
            raise HTTPException(401, 'Invalid API key')

    @api.get('/', include_in_schema=False)
    def home():
        return FileResponse(Path(__file__).parent / 'static' / 'index.html')

    @api.get('/health')
    def health():
        return {'status': 'ok', 'mode': settings.rag_mode}

    @api.get('/documents', dependencies=[Depends(authorize)])
    def documents():
        with lock:
            return {'documents': sorted({r['source'] for r in engine.chunks}), 'chunks': len(engine.chunks)}

    @api.post('/documents', dependencies=[Depends(authorize)])
    def ingest(doc: Document):
        try:
            with lock:
                count = engine.ingest(doc.source, doc.text)
            return {'source': doc.source, 'chunks': count}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, 'Document provider request failed; check provider configuration and quota.') from exc

    @api.post('/query', dependencies=[Depends(authorize)])
    def query(request: Query):
        try:
            with lock:
                return engine.answer(request.question, request.top_k)
        except httpx.HTTPError as exc:
            raise HTTPException(502, 'AI provider request failed; check provider configuration and quota.') from exc

    return api


class Document(BaseModel):
    source: str = Field(min_length=1, max_length=200, pattern=r'.*\S.*')
    text: str = Field(min_length=1, max_length=200_000, pattern=r'.*\S.*')


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=4000, pattern=r'.*\S.*')
    top_k: int = Field(default=5, ge=1, le=10)


app = create_app()
