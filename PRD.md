# Product Requirements Document (PRD): Dining & Fitness AI Concierge

> **Document Status**: Living Document  
> **Target Verticals**: Dining & Fitness  
> **Last Updated**: 2026-09-29  

---

## 1. Problem Statement

Booking dining reservations and fitness classes requires users to manually search fragmented availability, coordinate schedules with guests, manage OTP/payment steps, and re-enter preferences. Existing LLM-based booking prototypes often suffer from:
- **Hallucinations**: Claiming bookings are confirmed without API confirmation evidence.
- **Unbounded Execution**: Getting stuck in unreliable ReAct LLM tool loops or attempting unsupported requests (e.g. movies, flights, hotels).
- **Lack of Coordination**: Failing to check guest availability or notify guests upon booking confirmation.

---

## 2. Target Users

1. **Individual Consumers**: Users looking to quickly book restaurant tables or fitness classes based on personal preferences, location, and past history.
2. **Social Coordinators**: Users organizing group dinners or joint activities with named guests who need calendar conflict checks and guest email notifications.

---

## 3. Core Product Goals

- Build a **reliable, state-machine AI Concierge** using LangGraph to automate end-to-end dining and fitness bookings.
- Provide **human-in-the-loop safety** for pre-booking consent, payment inputs, and OTP handling.
- Maintain **long-term memory** for user profile preferences and booking history in SQLite.
- Seamlessly coordinate **Google Calendar, Google Contacts, and Gmail** via Composio for guest availability and invites.
- Enforce **strict vertical guardrails** (rejecting unsupported verticals like movies or hotels without memory pollution).

---

## 4. Key Functional & Technical Requirements

### 4.1 Supported Verticals & Scope Control
- **Supported Verticals**: Strictly **Dining** (restaurants, meals, tables) and **Fitness** (classes, studios, workouts).
- **Out-of-Scope Protection**: Immediately reject unsupported verticals (movies, flights, hotels, haircuts, concerts) and off-topic/malicious requests without triggering clarification prompts or polluting user preference memory.

### 4.2 State Machine Workflow ([agent.py](file:///Users/aldricpinto/Projects/opheliaLLM/agent.py))
- **Intent Extraction**: Parse natural language into structured JSON payloads (`operation`, `vertical`, `term`, `location`, `datetime_phrase`, `party_size`, `guest_name`).
- **Clarification**: Prompt user for missing required fields when necessary.
- **Venue Search & Slot Availability**: Search matching venues via Ophelia REST API and check time slot availability.
- **Human-in-the-Loop Interrupts**: Explicit user confirmation before booking execution, plus prompt interrupts for payment details and OTP verification.
- **Status Polling**: Async status reconciliation and settling windows for multi-step provider confirmations.

### 4.3 Long-Term Memory ([memory_store.py](file:///Users/aldricpinto/Projects/opheliaLLM/memory_store.py))
- **User Profile**: Persist relationship status, kids, pets, and dietary/activity preferences in SQLite (`ophelia_agent_memory.db`).
- **Booking History**: Log completed and attempted bookings to enable natural recall queries (e.g. *"What was the last Indian place I booked?"* or *"Book that place again"*).

### 4.4 Calendar & Guest Coordination ([composio_calendar.py](file:///Users/aldricpinto/Projects/opheliaLLM/composio_calendar.py))
- **Contact Resolution**: Resolve guest names to email addresses via Google Contacts.
- **Conflict Detection**: Check host and guest Google Calendars for schedule conflicts, proposing alternate times if busy.
- **Guest Notifications**: Send polished Gmail invitation emails with *"Add to Google Calendar"* links upon confirmation.

### 4.5 Rate-Limiting & Failure Protection
- Use **Redis-backed rate limiting** (`can_book`) and exponential cool-off wait windows to mitigate provider rate limits and Akamai blocks.

---

## 5. Success Metrics

| Metric | Target | Measurement |
| :--- | :--- | :--- |
| **Booking Success Rate** | > 90% | Ratio of consented booking attempts that reach `confirmed` status. |
| **Zero Hallucinations** | 100% | Zero confirmed summaries issued without verified `provider_booking_id` or confirmation code. |
| **Out-of-Scope Rejection** | 100% | Immediate rejection of unsupported requests with zero memory/preference pollution. |
| **Guest Invite Delivery** | > 95% | Successful Gmail invite dispatch when guest email is resolved. |

---

## 6. Risks & Security Guardrails

1. **False Confirmations & Hallucinated Codes**
   - *Guardrail*: Programmatic check (`has_confirmation_evidence`) requires explicit provider confirmation keys before summary generation.
2. **Sensitive Data Exposure**
   - *Guardrail*: Payment fields (card CVV, full numbers) and OTPs are held only in ephemeral state memory; **never** written to SQLite logs or DB tables.
3. **Memory Pollution from Bad Input**
   - *Guardrail*: Preference extraction is bypassed on unsupported verticals, failed requests, or memory queries.
4. **Provider Akamai / Rate-Limit Blocks**
   - *Guardrail*: Redis failure tracking (`increment_failures`) triggers automatic cool-off delays before re-attempting provider requests.
