# Clinical Companion

Local Docker boilerplate for the conversational agent: a Python **LangChain** agent that calls Hugging Face through the router. **Open WebUI** chats through `/v1`. The signed-in chat, **assistant-ui**, talks to the same agent through the LangGraph API on **companion-graph**. Login and access tokens come from **[fastapi_rbac](https://github.com/ingjohnguerrero/fastapi_rbac)**; the agent and companion-graph verify those JWTs locally.

No model weights are stored in the image. Inference goes out to Hugging Face (`HF_TOKEN`). RAG, citations, and extra knowledge bases are later sprints — this stack is the sandbox they plug into.

## Services

| Service | URL | Role |
| --- | --- | --- |
| `rbac` | http://localhost:8001/docs | Identity: `POST /auth/login`, users, roles |
| `rbac-admin` | http://localhost:3002 | Admin console for accounts and roles. Sign in with `ADMIN_USERNAME` (default `admin`) and `ADMIN_PASSWORD`. The email is not the login name. |
| `agent` | http://localhost:8000 | FastAPI + LangGraph (`/health` public; `/v1/*` needs Bearer JWT). In-memory threads for Open WebUI |
| `companion-graph` | http://localhost:2024/ok | Same agent image, `langgraph dev`. Durable threads for the signed-in chat |
| `assistant-ui` | http://localhost:3003 | Signed-in chat. Username and password; role `user` and role `admin` reach the same home |
| `open-webui` | http://localhost:3000 | Chat UI pointed at the agent (boots with a token from rbac) |
| `langfuse-web` | http://localhost:3001 | Local traces for each agent turn |

The agent is a LangChain tool loop with a content filter, PII checks, ebook and web search, a Tavily result filter, and a final safety check.

Authorize is token-first: rbac puts `sub`, `role`, and `permissions` in an HS256 JWT. The agent and companion-graph verify the same `JWT_SECRET`; they do not query the identity store on each run. The assistant-ui image does not contain `JWT_SECRET`. Its nginx proxies `/auth/` to rbac and `/api/` to companion-graph.

The signed-in chat is text only. A thread is owned by the JWT `sub` and stamped `channel` `assistant-ui`. Open WebUI chats stay on `/v1` and do not appear in that sidebar. Checkpoints and personal notes live on the `companion-state` volume (`/data/checkpoints.sqlite` and `/data/memories.sqlite`). That volume is not the Langfuse database. A message that starts with `Please remember that ` stores the remainder for that user. The same text is not stored twice, and the note is not a citation.

## Run

1. Copy the env file and set a Hugging Face token ([create one](https://huggingface.co/settings/tokens); it must start with `hf_`). Also set `JWT_SECRET`, `ADMIN_PASSWORD`, `TAVILY_API_KEY` (a [Tavily](https://tavily.com) key for web search), and the Langfuse values in `.env.example`:

   ```bash
   cp .env.example .env
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   openssl rand -hex 32
   openssl rand -base64 32
   ```

   Put the first string in `JWT_SECRET`. Put the hex string in `LANGFUSE_ENCRYPTION_KEY` (it must be 64 hex characters). Use the base64 strings for `LANGFUSE_NEXTAUTH_SECRET` and `LANGFUSE_SALT`. Change `ADMIN_PASSWORD` and `LANGFUSE_INIT_USER_PASSWORD`. Set `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` to `pk-lf-` plus a UUID and `LANGFUSE_INIT_PROJECT_SECRET_KEY` to `sk-lf-` plus a UUID, the same shape Langfuse generates. Database and MinIO passwords should use only letters, numbers, hyphens, and underscores.

2. Start the stack (first `rbac` build clones https://github.com/ingjohnguerrero/fastapi_rbac). The first Langfuse pull is large, and ClickHouse plus migrations often take a couple of minutes before the agent becomes healthy:

   ```bash
   docker compose up --build
   ```

3. Open the signed-in chat at http://localhost:3003 and sign in with `ADMIN_USERNAME` and `ADMIN_PASSWORD`. It waits until companion-graph answers at http://localhost:2024/ok. Open WebUI stays at http://localhost:3000; select the **clinical-companion** model. The admin console is at http://localhost:3002. Sign in there with the same username and password, not the admin email.

Langfuse starts only when `COMPOSE_PROFILES=langfuse` in `.env` (the example default). Leave that line empty to skip Langfuse and its databases, then stop any copies already running: `docker compose --profile langfuse stop`. With the profile set, Langfuse is at http://localhost:3001. Sign in with `LANGFUSE_INIT_USER_EMAIL` and `LANGFUSE_INIT_USER_PASSWORD`. After one chat completion, the project shows a trace for that turn: session id from `X-Session-Id` (or the request's thread id), user id from the JWT `sub`, and tag `clinical-companion`. Open WebUI uses one admin token, so every UI chat is that same subject. Traces stay in the Compose volumes.

On OrbStack, Compose domains are `https://<service>.clinical-companion.orb.local`. Open rbac at **https://rbac.clinical-companion.orb.local/docs** (there is no app at `/`). Host ports still work: rbac `http://localhost:8001/docs`.

Login is **username + password** (`ADMIN_USERNAME` / `ADMIN_PASSWORD` in `.env`), not email. In `/docs`, call `POST /auth/login`, then **Authorize** with `Bearer <access_token>`.

Optional: set `HF_MODEL` in `.env` (default `Qwen/Qwen3-4B-Instruct-2507`).

Open WebUI logs in to rbac once at start and uses that JWT as `OPENAI_API_KEY`. Default token lifetime in Compose is 24 hours (`ACCESS_TOKEN_EXPIRE_MINUTES`). Restart `open-webui` to mint a new one.

## Smoke test without the UI

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8001/health
curl -s http://localhost:2024/ok

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
assistant-ui/                      # signed-in chat; nginx proxies /auth and /api
rbac-admin/                        # accounts and roles console
agent/
  Dockerfile
  pyproject.toml          # entrypoint = app.main:app
  requirements.txt
  app/
    main.py               # FastAPI app, include_router
    auth.py               # verify rbac JWT
    dependencies.py       # session id, graph invoke
    config.py             # HF_TOKEN, HF_MODEL, JWT_SECRET
    routers/
      health.py           # GET /health
      openai.py           # /v1/models, /v1/chat/completions
    internal/
      graph.py            # companion graph; /v1 keeps in-memory threads
      llm.py              # ChatOpenAI on the Hugging Face router
```
