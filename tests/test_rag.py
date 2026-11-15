import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag import Engine, chunk_text, rrf


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Settings(data_dir=str(tmp_path), app_api_key='', rag_mode='demo')))


def test_chunking_overlap_and_boundaries():
    parts = chunk_text('a b c d e f g', size=4, overlap=1)
    assert parts == ['a b c d', 'd e f g', 'g']
    with pytest.raises(ValueError):
        chunk_text('hello', size=3, overlap=3)


def test_fusion_rewards_agreement_and_ignores_duplicate_ids():
    merged = rrf(['a', 'b', 'b'], ['b', 'c'])
    assert merged[0][0] == 'b'
    assert merged[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_api_ingest_query_replace_and_persist(client, tmp_path):
    assert client.post('/query', json={'question': 'incident'}).json()['sources'] == []
    response = client.post('/documents', json={'source': 'ops.md', 'text': 'Incident commanders send status updates every thirty minutes.'})
    assert response.status_code == 200
    result = client.post('/query', json={'question': 'Who sends incident updates?'}).json()
    assert result['sources'][0]['source'] == 'ops.md'
    assert '[1]' in result['answer']
    assert result['citation_check']['valid']
    client.post('/documents', json={'source': 'ops.md', 'text': 'Vacation requests require manager approval.'})
    engine = Engine(Settings(data_dir=str(tmp_path), rag_mode='demo'))
    assert len(engine.chunks) == 1
    assert 'Vacation' in engine.chunks[0]['text']


def test_validation_and_auth(tmp_path, client):
    assert client.post('/query', json={'question': 'hello', 'top_k': 0}).status_code == 422
    assert client.post('/documents', json={'source': 'x', 'text': '   '}).status_code == 422
    protected = TestClient(create_app(Settings(data_dir=str(tmp_path), app_api_key='test-secret', rag_mode='demo')))
    assert protected.get('/documents').status_code == 401
    assert protected.get('/documents', headers={'X-API-Key': 'test-secret'}).status_code == 200


def test_cloud_pipeline_contract(tmp_path, monkeypatch):
    engine = Engine(Settings(data_dir=str(tmp_path), rag_mode='cloud', anthropic_api_key='x',
                            cohere_api_key='x', pinecone_api_key='x', pinecone_host='https://example.test'))
    calls = []
    def post(url, body, headers):
        calls.append((url, body))
        if url.endswith('/embed'):
            return {'embeddings': [[0.1] * 1024 for _ in body['texts']]}
        if url.endswith('/query'):
            return {'matches': [{'id': engine.chunks[0]['id']}]}
        if url.endswith('/rerank'):
            return {'results': [{'index': 0, 'relevance_score': 0.95}]}
        if url.endswith('/messages'):
            return {'content': [{'type': 'text', 'text': 'The commander sends updates. [1]'}]}
        return {}
    monkeypatch.setattr(engine, 'post', post)
    engine.ingest('ops', 'The incident commander sends updates.')
    answer = engine.answer('Who sends incident updates?', 3)
    assert answer['citation_check']['valid']
    assert answer['sources'][0]['rerank_score'] == 0.95
    assert any(url.endswith('/vectors/upsert') for url, _ in calls)
    engine.ingest('ops', 'A different policy.')
    assert any(url.endswith('/vectors/delete') for url, _ in calls)


def test_invalid_generated_citation_is_flagged(tmp_path, monkeypatch):
    engine = Engine(Settings(data_dir=str(tmp_path), rag_mode='demo'))
    engine.ingest('ops', 'Incident commander sends updates.')
    engine.s.rag_mode = 'cloud'
    monkeypatch.setattr(engine, 'search', lambda *args: [{'text': 'Incident commander sends updates.', 'source': 'ops'}])
    monkeypatch.setattr(engine, 'post', lambda *args: {'content': [{'type': 'text', 'text': 'Updates. [99]'}]})
    result = engine.answer('incident?', 5)
    assert result['citation_check'] == {'valid': False, 'invalid_ids': [99]}
