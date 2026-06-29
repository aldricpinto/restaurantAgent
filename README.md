# Ophelia LLM Agent MVP

Direct REST AI agent prototype for booking through the Ophelia API.

The agent uses:

- LangGraph for the state machine and interrupt/resume flow
- Groq/LangChain for LLM intent extraction and grounded summaries
- Direct HTTPS calls to the Ophelia REST API
- SQLite + SQLAlchemy for local user profile and memory


## 1. Setup

From this folder:

```bash
cd /Users/aldricpinto/Projects/opheliaLLM
```

Install dependencies with `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Environment variables

Create a `.env` file in the project root.

### Required

```env
OPHELIA_API_KEY=your_ophelia_api_key
GROQ_API_KEY=your_groq_api_key
```

`OPHELIA_API_KEY` is used by `ophelia_client.py` for authenticated Ophelia REST calls.

`GROQ_API_KEY` is used by `ChatGroq` through LangChain.

### Optional

```env
# Defaults to https://api.opheliaos.com/v1
OPHELIA_BASE_URL=https://api.opheliaos.com/v1

# MVP local user identity. Production should pass authenticated user/session context.
OPHELIA_ORG_ID=demo_org
OPHELIA_USER_ID=demo_user

# Defaults to llama-3.3-70b-versatile
GROQ_MODEL=llama-3.3-70b-versatile

# SQLite memory store. Defaults to ophelia_agent_memory.db
OPHELIA_MEMORY_DB=ophelia_agent_memory.db

# Full SQLAlchemy DB URL. If set, overrides OPHELIA_MEMORY_DB.
OPHELIA_MEMORY_DATABASE_URL=sqlite:///ophelia_agent_memory.db

# Currently present for local experimentation; not required by the core MVP flow.
REDIS_URL=redis://localhost:6379/0
```

### Optional LangSmith tracing

Use this only if you want LangChain/LangGraph traces in LangSmith:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=opheliaTest
LANGCHAIN_CALLBACKS_BACKGROUND=true
```

Do not commit real API keys. If a key is shared in chat or committed accidentally, rotate it.

## 3. Run the CLI agent

Interactive mode:

```bash
uv run agent.py
```

Or:

```bash
python agent.py
```

You can also pass the first user request as command-line text:

```bash
uv run agent.py "Book me an Italian restaurant in SoHo tomorrow at 8 PM for 2 people"
```

## 4. Example flows

Dining booking:

```text
Book me an Indian restaurant near Times Square tomorrow at 8 PM for 2 people
```

Fitness booking:

```text
Book me a pilates class near SoHo tomorrow morning
```

Memory recall:

```text
What was the last Indian place I booked?
```

Follow-up from memory:

```text
Book that place again tomorrow at 9 PM for 2 people
```

## 5. Local files generated

The app may create/update:

- `ophelia.log` — local logs
- `workflow.png` — generated LangGraph workflow image
- `ophelia_agent_memory.db` — local SQLite memory database

These files are useful for demo/debugging but should be reviewed before committing.

## 6. Safety notes

- Payment details, CVV, OTP, Mindbody password, and API keys should not be stored in SQLite memory.
- Ophelia booking status is treated as authoritative.
- The final summary should not claim a booking is confirmed unless Ophelia returns a confirmed status with confirmation evidence.
- Customer/payment/password fields are collected only immediately before create/continue calls.

