# Clinical Companion

Local Docker boilerplate for the conversational agent: a Python **LangGraph** service that calls Hugging Face models through **smolagents**, plus **Open WebUI** as the chat client.

No model weights are stored in the image. Inference goes out to Hugging Face (`HF_TOKEN`). RAG, citations, and extra knowledge bases are later sprints — this stack is the sandbox they plug into.

## Services

| Service | URL | Role |
| --- | --- | --- |
| `agent` | http://localhost:8000 | FastAPI + LangGraph (`/health`, `/v1/models`, `/v1/chat/completions`) |
| `open-webui` | http://localhost:3000 | Chat UI pointed at the agent |

The graph is `START → retrieve (stub) → generate → END`. `retrieve` returns empty context for now.

## Run

1. Copy the env file and set a Hugging Face token ([create one](https://huggingface.co/settings/tokens); it must start with `hf_`):

   ```bash
   cp .env.example .env
   ```

2. Start both containers:

   ```bash
   docker compose up --build
   ```

3. Open http://localhost:3000 and select the **clinical-companion** model.

Optional: set `HF_MODEL` in `.env` (default `Qwen/Qwen2.5-72B-Instruct`).

## Smoke test without the UI

```bash
curl -s http://localhost:8000/health

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clinical-companion",
    "messages": [{"role": "user", "content": "What is a balanced breakfast pattern?"}]
  }'
```

## Layout

Package layout follows [FastAPI bigger applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/): `app.main:app`, `APIRouter` modules, shared `dependencies`, internal graph/LLM code.

```
.env.example
docker-compose.yml
agent/
  Dockerfile
  pyproject.toml          # entrypoint = app.main:app
  requirements.txt
  app/
    main.py               # FastAPI app, include_router
    dependencies.py       # session id, graph invoke
    config.py             # HF_TOKEN, HF_MODEL
    routers/
      health.py           # GET /health
      openai.py           # /v1/models, /v1/chat/completions
    internal/
      graph.py            # LangGraph stub
      llm.py              # smolagents InferenceClientModel
```
