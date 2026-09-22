# Taking Ophelia Intelligence To The Next Level With Composio

This document frames the next step after the MVP demo: keep Ophelia as the booking execution engine, and use Composio as the surrounding intelligence/action layer that lets the agent make better decisions, ask fewer questions, and complete the workflow around the booking.

## Core Idea

The current prototype proves the hardest core loop:

```text
Natural language request
  -> intent parsing
  -> Ophelia REST API search
  -> booking execution
  -> OTP/payment continuation
  -> confirmed status
```

The next level is not replacing this. The next level is giving the agent more context before it calls Ophelia, and more useful actions after Ophelia completes the booking.

In one sentence:

> Composio gives the Ophelia agent hands and eyes across the user's apps, while Ophelia remains the source of truth for booking execution.

## Visual Architecture

Today:

```text
User
  -> Ophelia LLM Agent
  -> Ophelia REST API
  -> Provider booking
  -> Confirmation
```

Next:

```text
                       Calendar
                          |
Email / Gmail ------------|
                          |
Contacts -----------------|        Preference Memory
                          |              |
Slack / Teams ------------|              |
                          v              v
User --------------> Ophelia Intelligence Layer
                          |
                          v
                  Ophelia REST API
                          |
                          v
             Dining / Fitness / Entertainment
                          |
                          v
             Confirmation + follow-up actions
                          |
                          v
          Calendar event / message / note / CRM update
```

The agent becomes less like a command parser and more like a personal booking copilot.

## Division Of Responsibility

The architecture should stay clean:

```text
Composio: external app context and actions
Agent: reasoning, policy, decision-making, state
Ophelia API: search, availability, booking, status, confirmation
```

Composio should not become the booking layer. Ophelia already owns that. Composio should help the agent decide what to ask, what to infer, who to notify, and what to do after a booking is confirmed.

## Why This Matters

The current agent can answer:

```text
"Book sushi in SoHo tomorrow at 8 PM for 2."
```

A Composio-augmented Ophelia agent can handle:

```text
"Plan dinner with Sam after work on Friday."
```

That request has hidden requirements:

- Who is Sam?
- What calendar window works?
- Where will the user be after work?
- What neighborhoods are reasonable?
- What food does the user usually like?
- Should Sam get a calendar invite?
- Should the user get a reminder?
- Should this be remembered for next time?

Composio gives the agent ways to answer those questions through connected tools instead of making the user type everything manually.

## Intelligence Upgrade

### 1. Fewer Questions

Without Composio:

```text
What date?
What time?
Which neighborhood?
How many people?
Who should I invite?
Should I add it to your calendar?
```

With Composio:

```text
I found that you and Sam are both free Friday from 7:30 to 9:30.
You usually prefer downtown Italian or sushi for dinner.
I can book a table for two near SoHo at 8:00 or 8:15.
```

The user still stays in control, but the agent arrives prepared.

### 2. Better Decision-Making

Composio can provide context from:

- Calendar availability
- Email confirmations
- Contact records
- Slack messages
- Notes or CRM data
- Previous plans
- Task lists

The agent can combine that context with Ophelia booking capabilities:

```text
Calendar says the user is free after 7:30.
Memory says the user likes Italian and SoHo.
Contact lookup says Sam's email is sam@example.com.
Ophelia search returns available restaurants.
Agent recommends the best slot and asks for consent.
Ophelia executes booking.
Composio creates the calendar event and notifies Sam.
```

### 3. Full Workflow Completion

Booking is only one part of the real user journey.

After Ophelia confirms, the agent can:

- Create a calendar event.
- Invite guests.
- Send a confirmation message.
- Add the address and confirmation code.
- Create a reminder.
- Save the user's preference.
- Log the event to a CRM or personal notes system.

This makes the experience feel finished.

## Concrete Example: Dinner Planning

User:

```text
Plan dinner with Sam this Friday after work.
```

Agent flow:

```text
1. Interpret high-level intent:
   "User wants a dining booking, likely for two people."

2. Use Composio/context tools:
   - Check user's calendar for Friday.
   - Find Sam in contacts.
   - Optionally check Sam's availability if shared.
   - Read local memory/preferences.

3. Convert ambiguous intent into a concrete booking plan:
   - vertical: dining
   - term: Italian or sushi based on preference
   - location: SoHo or near after-work location
   - datetime: Friday at 8 PM
   - party_size: 2

4. Call Ophelia REST API:
   - POST /v1/venues/search
   - show venue choices
   - POST /v1/bookings after consent
   - POST /continue if OTP/payment is required

5. Use Composio after confirmation:
   - create calendar event
   - invite Sam
   - send confirmation
   - save preference/memory
```

Result:

```text
Dinner is confirmed for Friday at 8:00 PM at L'Artusi.
I added it to your calendar and invited Sam.
Confirmation: bkg_...
```

## Concrete Example: Fitness Booking

User:

```text
Book me a pilates class sometime tomorrow morning.
```

With context tools:

- Calendar shows meetings from 9:30 to 11:00.
- Memory says the user prefers SoHo and morning classes.
- Agent searches for pilates around 7:30-9:00 or after 11:00.
- Ophelia executes the Mindbody booking.
- After confirmation, agent adds it to the calendar.

This makes the agent proactive without becoming reckless.

## Concrete Example: Entertainment

User:

```text
Find tickets for The Weeknd this weekend and invite Maya.
```

With context tools:

- Calendar finds open weekend evenings.
- Contact lookup finds Maya.
- Ophelia searches StubHub through the entertainment vertical.
- Agent shows ticket sections/prices from Ophelia metadata.
- User picks one.
- Ophelia executes ticket purchase flow.
- Agent emails or messages Maya after confirmation.

## What Is Now Implemented

This prototype now includes Composio inside the LangGraph agent for a real Google Calendar use case.

The demo request is:

```text
Book a dinner for Tony Stark and me at 8 PM at an Italian place near SoHo and add it to calendar
```

The graph now does this:

```text
extract_intent
  -> resolve_people_context
  -> check_calendar_availability
  -> propose_alternate_time when needed
  -> Ophelia search and booking flow
  -> create_calendar_invite after confirmed booking
  -> grounded final summary
```

Real Composio responsibilities:

- connect Aldric's Google Calendar and Google Contacts through `uv run agent.py --connect-calendar`
- resolve Tony Stark's email through Google Contacts before using any env fallback when `COMPOSIO_CONTACTS_AUTH_CONFIG_ID` is configured
- check Aldric's calendar before booking when calendar coordination is requested
- check Tony Stark's calendar only if visible/shared through Aldric's Google account
- send a Gmail guest notification after Ophelia returns `confirmed`
- notify Tony through a guest-facing Gmail message

Environment knobs:

```env
COMPOSIO_API_KEY=...
COMPOSIO_ENABLE_CALENDAR=true
COMPOSIO_USER_ID=aldric
COMPOSIO_CACHE_DIR=/private/tmp/ophelia-composio-cache
OPHELIA_USER_NAME=Aldric
OPHELIA_USER_EMAIL=aldric@example.com
COMPOSIO_CONTACTS_TOOLKIT=googlecontacts

# Optional fallback only if Google Contacts cannot resolve the guest.
TONY_STARK_EMAIL=tony@example.com
GROQ_MODEL=openai/gpt-oss-120b
```

Important guardrail: if the calendar step fails, the agent keeps the Ophelia booking status separate from the calendar failure. It never turns a calendar failure into a failed booking.

## How Composio Fits Production

Composio would become the adapter layer for connected apps:

```text
Google Calendar / Outlook
  -> availability lookup
  -> event creation
  -> reminders

Gmail / Slack / Teams
  -> confirmations
  -> guest coordination
  -> follow-up messages

Contacts
  -> resolve "Sam" to email/phone
  -> invite the right person

Notion / CRM
  -> log bookings
  -> record client preferences
  -> update trip or event plans
```

The agent should call Composio tools for context and workflow, then call Ophelia for booking.

## Product Vision

Ophelia can become more than:

```text
"Book X at Y time."
```

It can become:

```text
"Handle this plan for me."
```

The difference is context.

The booking API answers:

```text
Can this reservation be made?
```

The intelligence layer answers:

```text
What should we book, when, where, with whom, and what should happen afterward?
```

Composio helps with the second question. Ophelia owns the first.

## Guardrails

The agent should still follow strict rules:

- Do not create a booking without user consent.
- Do not claim confirmation unless Ophelia returns `confirmed`.
- Do not store card, CVV, OTP, password, or API keys in memory.
- Use Ophelia as the authoritative booking status source.
- Use connected app data only with explicit user authorization.
- Keep context gathering separate from booking execution.

## Technical Story For The Call

If they ask "why Composio?":

> Because booking intent often depends on information outside the booking API: calendar, contacts, messages, and workflow destinations. Composio gives the agent a safe integration layer for that surrounding context.

If they ask "does this replace Ophelia?":

> No. Ophelia remains the booking system of record. Composio only helps before and after the booking.

If they ask "what would you build first?":

> I would build a mock-backed context planning demo: calendar windows, contact lookup, preference memory, Ophelia search/booking, then calendar-event creation. It proves the architecture without waiting on OAuth.

If they ask "how does this make the agent smarter?":

> It lets the agent infer missing details from authorized context instead of asking the user every question. That creates a concierge-like experience while keeping the real booking execution deterministic through Ophelia.

## Proposed Wednesday Close

> The MVP proved that natural language can drive a real Ophelia booking through direct REST. The next step is to make the agent context-aware. Composio can give the agent access to calendar, contacts, messages, and workflow tools, while Ophelia remains the execution layer. I can build a simple context-aware demo that turns "plan dinner with Sam Friday after work" into calendar-aware search, Ophelia booking, and a follow-up calendar invite.

