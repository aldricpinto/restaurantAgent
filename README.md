# Ophelia LLM Agent MVP

Direct REST AI agent prototype for booking through the Ophelia API.

The agent uses:

- LangGraph for the state machine and interrupt/resume flow
- Groq/LangChain for LLM intent extraction and grounded summaries
- Direct HTTPS calls to the Ophelia REST API
- SQLite + SQLAlchemy for local user profile and memory


## 1. Prerequisites

You need:

- Python 3.11+
- `uv` or `pip`
- Ophelia API key
- Groq API key
- Optional: Composio API key for Google Contacts/Calendar/Gmail

## 2. Setup

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

## 3. Environment variables

Create a `.env` file in the project root. The easiest path is:

```bash
cp .env.example .env
```

Then fill in your real keys.

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

# Defaults to openai/gpt-oss-120b in this prototype
GROQ_MODEL=openai/gpt-oss-120b

# SQLite memory store. Defaults to ophelia_agent_memory.db
OPHELIA_MEMORY_DB=ophelia_agent_memory.db

# Full SQLAlchemy DB URL. If set, overrides OPHELIA_MEMORY_DB.
OPHELIA_MEMORY_DATABASE_URL=sqlite:///ophelia_agent_memory.db

# Currently present for local experimentation; not required by the core MVP flow.
REDIS_URL=redis://localhost:6379/0

# Optional Composio context/action integration
# Enables Google Contacts lookup, Google Calendar availability, and Gmail guest notifications.
COMPOSIO_API_KEY=your_composio_api_key
COMPOSIO_ENABLE_CALENDAR=true
COMPOSIO_USER_ID=demo_user
COMPOSIO_CACHE_DIR=/private/tmp/ophelia-composio-cache
COMPOSIO_CALENDAR_TOOLKIT=googlecalendar
COMPOSIO_CONTACTS_TOOLKIT=googlecontacts
COMPOSIO_GMAIL_TOOLKIT=gmail

# Required if your Composio toolkit cannot auto-create auth configs.
# Create these in the Composio dashboard from your Google OAuth app/client.
COMPOSIO_CALENDAR_AUTH_CONFIG_ID=ac_...
COMPOSIO_CONTACTS_AUTH_CONFIG_ID=ac_...
COMPOSIO_GMAIL_AUTH_CONFIG_ID=ac_...

# Human-readable identity for guest notifications.
OPHELIA_USER_NAME=Aldric
OPHELIA_USER_EMAIL=aldric@example.com

# Optional fallback only if Google Contacts cannot resolve the guest.
TONY_STARK_EMAIL=tony@example.com
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

## 4. Run the CLI agent

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

## 5. Example flows

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

## 6. Local files generated

The app may create/update:

- `ophelia.log` — local logs
- `workflow.png` — generated LangGraph workflow image
- `ophelia_agent_memory.db` — local SQLite memory database

These files are useful for demo/debugging but should be reviewed before committing.

## 7. Safety notes

- Payment details, CVV, OTP, Mindbody password, and API keys should not be stored in SQLite memory.
- Ophelia booking status is treated as authoritative.
- The final summary should not claim a booking is confirmed unless Ophelia returns a confirmed status with confirmation evidence.
- Customer/payment/password fields are collected only immediately before create/continue calls.


## 8. Optional Composio Google Flow

This prototype can use Composio inside the LangGraph flow for calendar-aware booking coordination. Ophelia still executes all search and booking operations through direct REST. Composio is used only around the booking:

- Google Contacts resolves guest names to email addresses.
- Google Calendar checks availability for the current user and visible/shared guest calendars.
- Gmail sends the guest a polished booking notification after Ophelia confirms the booking.

The agent does not create a duplicate host calendar event after booking. OpenTable/Gmail can handle the organizer's own event; the agent sends the guest a notification with an Add to Google Calendar link.

### Composio setup

1. Create or select a Composio project and set `COMPOSIO_API_KEY`.
2. In Google Cloud, create an OAuth app/client for the developer/app owner.
3. Enable the Google APIs needed by the OAuth project:
   - Google Calendar API
   - Google People API
   - Gmail API
4. In Composio, create auth configs for Google Calendar, Google Contacts, and Gmail if Composio does not auto-create them. Put those `ac_...` IDs in `.env`.
5. If the Google OAuth app is in testing mode, add every demo Google account as a test user in Google Cloud.
6. Connect the actual end-user Google account once:

```bash
uv run agent.py --connect-calendar
```

Despite the command name, it attempts to connect Calendar, Contacts, and Gmail.

Then run a calendar-aware request:

```bash
uv run agent.py "Book a business dinner for me and Tony Stark near Times Square tomorrow at 8 PM at an Indian place"
```

For a request like "Tony Stark and me", the graph resolves Tony through Google Contacts first when `COMPOSIO_CONTACTS_AUTH_CONFIG_ID` is configured. The `TONY_STARK_EMAIL` style env var is only a fallback for demos or missing contact permissions. If the requested time conflicts, the agent interrupts with a friendly alternate time before booking. After Ophelia returns a confirmed booking, the agent sends the guest a Gmail notification with the booking details.

Troubleshooting:

- If Google says the app is blocked because it has not completed verification, add the connecting Google account as an OAuth test user or complete Google app verification.
- If Google Contacts returns `People API has not been used ... or it is disabled`, enable the Google People API in the Google Cloud project that owns your OAuth client, then wait a few minutes and reconnect/retry.
- If Gmail cannot send, make sure the Gmail API is enabled and the connected Google account has granted Gmail permission through Composio.
- Contact lookup, calendar availability, and Gmail guest notification actions use direct Composio router execution, not an LLM tool loop, to avoid sending large Composio tool schemas to Groq.
