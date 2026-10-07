# Clinical Companion

Local Docker boilerplate for the conversational agent: a Python **LangGraph** service that calls Hugging Face models through **smolagents**, plus **Open WebUI** as the chat client. Login and access tokens come from **[fastapi_rbac](https://github.com/ingjohnguerrero/fastapi_rbac)**; the agent verifies those JWTs before serving `/v1` routes.

No model weights are stored in the image. Inference goes out to Hugging Face (`HF_TOKEN`). RAG retrieval uses a **Weaviate** vector store; embeddings use the Hugging Face Inference API (same stack as smolagents). The agent answers only from uploaded PDF documents and states when information is unavailable.

## Services

| Service | URL | Role |
| --- | --- | --- |
| `rbac` | http://localhost:8001/docs | Identity: `POST /auth/login`, users, roles |
| `agent` | http://localhost:8000 | FastAPI + LangGraph (`/health` public; `/v1/*` needs Bearer JWT) |
| `weaviate` | http://localhost:8080 | Vector store for RAG citation-aware retrieval |
| `ingest` | (one-shot) | Embeds PDF knowledge base into Weaviate |
| `open-webui` | http://localhost:3000 | Chat UI pointed at the agent (boots with a token from rbac) |

The LangGraph multi-source cascade is:

`guard → classify → rag_search → rag_grade → (rag_answer | web_search → web_answer | general)`

`rag_search` performs a **hybrid search** (semantic vector + BM25 keyword) over Weaviate, then `rag_grade` evaluates whether the retrieved passages are sufficient. If they are, `rag_answer` elaborates from the book with inline citations (author, title, page, paragraph). If not, the cascade falls through to web search (`Tavily`) or general knowledge. The answer prompt enforces **strict grounding**: it only uses the retrieved text and states when information is unavailable.

Authorize is token-first: rbac puts `sub`, `role`, and `permissions` in an HS256 JWT. The agent verifies the same `JWT_SECRET`; it does not query the identity store.

## Run

1. Copy the env file and set a Hugging Face token ([create one](https://huggingface.co/settings/tokens); it must start with `hf_`). Also set `JWT_SECRET`, `ADMIN_PASSWORD`, and `TAVILY_API_KEY` (a [Tavily](https://tavily.com) key for web search):

   ```bash
   cp .env.example .env
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

   Put the printed string in `JWT_SECRET`. Change `ADMIN_PASSWORD`.

2. Start the stack (first `rbac` build clones https://github.com/ingjohnguerrero/fastapi_rbac):

   ```bash
   docker compose up --build
   ```

3. Open http://localhost:3000 and select the **clinical-companion** model.

On OrbStack, Compose domains are `https://<service>.clinical-companion.orb.local`. Open rbac at **https://rbac.clinical-companion.orb.local/docs** (there is no app at `/`). Host ports still work: rbac `http://localhost:8001/docs`.

Login is **username + password** (`ADMIN_USERNAME` / `ADMIN_PASSWORD` in `.env`), not email. In `/docs`, call `POST /auth/login`, then **Authorize** with `Bearer <access_token>`.

Optional: set `HF_MODEL` in `.env` (default `Qwen/Qwen2.5-72B-Instruct`).

Optional: set `HF_EMBEDDING_MODEL` in `.env` (default `intfloat/multilingual-e5-large`).

The `ingest` service runs once at startup to embed the PDF in `agent/knowledge/` into Weaviate.  To re-ingest after changing documents:

```bash
docker compose run ingest
```

Open WebUI logs in to rbac once at start and uses that JWT as `OPENAI_API_KEY`. Default token lifetime in Compose is 24 hours (`ACCESS_TOKEN_EXPIRE_MINUTES`). Restart `open-webui` to mint a new one.

## Smoke test without the UI

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8001/health

TOKEN=$(curl -s http://localhost:8001/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"'"$ADMIN_PASSWORD"'"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "model": "clinical-companion",
    "messages": [{"role": "user", "content": "What is a balanced breakfast pattern?"}]
  }'
```

Use the username/password from `.env`. A request without a valid Bearer token on `/v1/*` returns **401**.

Interactive rbac contract: http://localhost:8001/docs

## Layout

Package layout follows [FastAPI bigger applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/): `app.main:app`, `APIRouter` modules, shared `dependencies`, internal graph/LLM code.

```
.env.example
docker-compose.yml
scripts/open-webui-entrypoint.py   # mint JWT, then start Open WebUI
agent/
  Dockerfile
  pyproject.toml          # entrypoint = app.main:app
  requirements.txt
  app/
    main.py               # FastAPI app, include_router
    auth.py               # verify rbac JWT
    config.py             # HF_TOKEN, HF_MODEL, HF_EMBEDDING_MODEL, WEAVIATE_*
    dependencies.py       # session id, graph invoke
    routers/
      health.py           # GET /health
      openai.py           # /v1/models, /v1/chat/completions
    internal/
      graph.py            # LangGraph multi-source cascade
      llm.py              # smolagents InferenceClientModel
      book.py             # Weaviate-backed BookIndex, citation formatting
      weaviate_schema.py  # Collection schema with rich metadata
      weaviate_client.py  # Lazy Weaviate client factory
      embeddings.py       # Hugging Face inference embeddings
      ingest.py           # PDF ingestion pipeline
    scripts/
      ingest_docs.py      # CLI: embed PDFs into Weaviate
```
