# Product Requirements Document: Event Ticket Price Comparison
**Feature:** Ticketmaster & StubHub Price Comparison Agent  
**Product:** AI Trip Planner - Entertainment Enhancement  
**Author:** Principal Product Manager | **Date:** October 30, 2025 | **Status:** Proposal

---

## 1. Problem Statement
Travelers planning trips to new cities often want to attend concerts, sports events, or shows, but struggle to find best ticket prices across multiple platforms. Manual comparison between Ticketmaster (primary market) and StubHub (secondary market) is time-consuming and leads to overpaying or missing out on events that fit their budget and interests.

## 2. Solution Overview
Extend the AI Trip Planner's multi-agent architecture with an **Events Agent** that fetches and compares ticket prices from Ticketmaster and StubHub, integrating seamlessly into the existing trip planning workflow. The agent will surface event recommendations that match the user's destination, travel dates, interests, and budget.

## 3. User Stories
- **As a traveler**, I want to see upcoming events (concerts, sports, theater) in my destination during my travel dates so I can plan entertainment activities.
- **As a budget-conscious user**, I want to compare ticket prices between Ticketmaster and StubHub so I can find the best deals.
- **As a music/sports enthusiast**, I want personalized event recommendations based on my stated interests so I don't miss relevant events.
- **As a trip planner**, I want event costs integrated into my overall budget breakdown so I can make informed spending decisions.

## 4. Success Metrics
- **Engagement:** 40%+ of users interact with event recommendations within 2 weeks of launch
- **Conversion:** 15%+ of users click through to ticket purchase links
- **Satisfaction:** 4.2+ star rating on event recommendations (in-app feedback)
- **Performance:** Event search completes in <3 seconds, price comparison in <5 seconds

## 5. Technical Architecture

### 5.1 Agent Integration
**New Agent:** `events_agent()`  
**Position:** Runs in parallel with existing Research, Budget, and Local agents  
**Tools:** `search_events()`, `compare_ticket_prices()`, `filter_by_budget()`

```
┌─────────── LangGraph Parallel Execution ───────────┐
│  Research Agent │ Budget Agent │ Local Agent │ Events Agent  │
└────────────────┴──────────────┴─────────────┴──────────────┘
                            ↓
                    Itinerary Agent
                 (synthesizes all inputs)
```

### 5.2 API Integrations
**Primary APIs:**
- **Ticketmaster Discovery API** (Primary Market)
  - Endpoint: `/discovery/v2/events.json`
  - Rate Limit: 5,000 calls/day (free tier)
  - Data: Official tickets, pricing, venue, dates
  
- **StubHub API** (Secondary/Resale Market)
  - Endpoint: `/search/v3/events`
  - Rate Limit: Contact for pricing (likely paid)
  - Data: Resale tickets, market pricing, inventory

**Fallback Strategy:** (following existing pattern in `main.py`)
1. Try Ticketmaster → Try StubHub → Use LLM fallback with cached data
2. Cache responses for 6 hours (events change less frequently than weather)
3. Show "prices as of [timestamp]" to set user expectations

### 5.3 New Tools (following existing patterns)

```python
@tool
def search_events(destination: str, start_date: str, end_date: str, 
                  category: Optional[str] = None) -> str:
    """Search for events in destination during travel dates."""
    # Ticketmaster API call with graceful degradation
    query = f"{destination} events {start_date} to {end_date} {category or ''}"
    summary = _search_api_ticketmaster(query)
    if summary:
        return _format_events(summary)
    return _llm_fallback(f"List major events in {destination} during {start_date}-{end_date}")

@tool
def compare_ticket_prices(event_name: str, destination: str) -> str:
    """Compare prices for an event across Ticketmaster and StubHub."""
    tm_price = _fetch_ticketmaster_price(event_name, destination)
    sh_price = _fetch_stubhub_price(event_name, destination)
    return _format_price_comparison(event_name, tm_price, sh_price)
```

### 5.4 State Extensions
```python
class TripState(TypedDict):
    # ... existing fields ...
    events: Optional[str]  # Events agent output
    event_recommendations: List[Dict[str, Any]]  # Structured event data
```

## 6. User Experience Flow

### 6.1 Input Enhancement
Extend `TripRequest` model:
```python
class TripRequest(BaseModel):
    # ... existing fields ...
    include_events: Optional[bool] = True  # Feature flag
    event_categories: Optional[List[str]] = None  # ["sports", "music", "theater"]
```

### 6.2 Output Format
Events Agent returns markdown-formatted recommendations:
```
🎭 Events During Your Trip (Nov 15-20, 2025)

1. Taylor Swift - Eras Tour
   📍 SoFi Stadium, Los Angeles
   📅 Nov 17, 8:00 PM
   💰 Ticketmaster: $350-$850 | StubHub: $420-$1,200
   💡 Best Value: Ticketmaster (official) | Get Tickets →

2. Lakers vs. Warriors
   📍 Crypto.com Arena
   📅 Nov 19, 7:30 PM
   💰 Ticketmaster: $85-$450 | StubHub: $75-$380
   💡 Best Value: StubHub (20% cheaper) | Get Tickets →
```

## 7. Implementation Plan (4 Weeks)

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | API Setup & Authentication | Ticketmaster & StubHub API keys, rate limit testing, error handling |
| 2 | Events Agent Development | `events_agent()` function, tools, state integration, parallel execution |
| 3 | Price Comparison Logic | Comparison algorithm, caching, fallback strategies, affiliate link integration |
| 4 | Testing & Launch | Integration testing, observability (Arize traces), 10% rollout, full launch |

## 8. API Cost Analysis

| Service | Free Tier | Paid Tier | Est. Monthly Cost |
|---------|-----------|-----------|-------------------|
| Ticketmaster Discovery API | 5,000 calls/day | N/A (sufficient) | $0 |
| StubHub API | TBD | ~$500/month (est.) | $500 |
| **Total Estimated** | | | **$500/month** |

**Cost Optimization:**
- Aggressive caching (6-hour TTL for events, 1-hour for pricing)
- Only fetch StubHub prices when Ticketmaster shows >$100 tickets (high-value comparison)
- Batch requests for multiple events in same city

## 9. Risks & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| StubHub API costs exceed budget | High | Medium | Start with Ticketmaster-only MVP; add StubHub in Phase 2 if ROI proven |
| API rate limits during high traffic | Medium | Medium | Implement Redis caching, queue system for non-critical requests |
| Inaccurate pricing (sold out, price changes) | Medium | High | Show timestamps, disclaimers, link directly to ticketing sites for verification |
| Legal/compliance issues with price scraping | High | Low | Use official APIs only, comply with ToS, include affiliate disclosures |

## 10. Open Questions & Decisions Needed
1. **Revenue model:** Affiliate commissions from ticket sales, or pure feature add-on?
2. **StubHub alternative:** Should we evaluate Vivid Seats or SeatGeek as lower-cost alternatives?
3. **International support:** Start with US-only or support international destinations from Day 1?
4. **User preferences:** Should users opt-in to events, or include by default with opt-out?

---

## Next Steps
1. **Technical spike** (3 days): Validate Ticketmaster & StubHub API access, test response times
2. **Design review** (1 day): Finalize event card UI in itinerary output
3. **Go/No-Go decision** (1 day): Present findings to leadership; approve $500/month StubHub budget
4. **Implementation** (4 weeks): Follow timeline above

