# Health Insurance Marketplace - Product Requirements Document

## Overview

**Product Name:** HealthMatch AI

**Description:** An AI-powered marketplace that helps consumers shop for health insurance by analyzing their needs, comparing plans, and providing personalized recommendations—all through a simple conversational interface.

**Target Users:** Individuals and families shopping for health insurance during open enrollment or qualifying life events (job change, marriage, turning 26, etc.)

---

## Problem Statement

Shopping for health insurance is overwhelming. Consumers face hundreds of plan options with complex terminology (deductibles, coinsurance, out-of-pocket maximums), confusing provider networks, and no easy way to compare plans based on their actual healthcare needs. Most end up choosing based on premium alone, often resulting in unexpected costs or inadequate coverage.

**HealthMatch AI** solves this by acting as a personal insurance advisor—gathering user needs, researching plans, analyzing costs, checking provider networks, and recommending the best options in seconds.

---

## Goals

1. Reduce time to find the right health insurance plan from hours to minutes
2. Help users understand true costs (not just premiums) based on their expected healthcare usage
3. Ensure recommended plans include their preferred doctors and hospitals
4. Increase user confidence in their insurance decision

## Non-Goals

- Selling insurance directly (we refer users to carriers/exchanges)
- Providing medical advice
- Handling claims or customer service for insurers

---

## Agent Architecture

### Multi-Agent System (LangGraph)

| Agent | Purpose | Tools |
|-------|---------|-------|
| **Needs Analyzer** | Understand user's health situation, medications, preferred doctors | `parse_health_profile`, `medication_lookup` |
| **Plan Researcher** | Find eligible plans based on location, income, enrollment period | `search_plans`, `check_eligibility`, `plan_details` |
| **Cost Analyzer** | Calculate true annual costs based on expected usage | `estimate_costs`, `compare_premiums`, `subsidy_calculator` |
| **Network Checker** | Verify if user's doctors/hospitals are in-network | `provider_search`, `network_lookup`, `facility_check` |
| **Recommendation Agent** | Synthesize all inputs and recommend top 3 plans with reasoning | Uses outputs from all agents |

### Execution Flow

```
                        User Input
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Needs   │  │   Plan   │  │ Network  │
        │ Analyzer │  │Researcher│  │ Checker  │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
              │             │             │
              └─────────────┼─────────────┘
                            │
                    ┌───────▼───────┐
                    │ Cost Analyzer │
                    └───────┬───────┘
                            │
                   ┌────────▼────────┐
                   │  Recommendation │
                   │      Agent      │
                   └────────┬────────┘
                            │
                    Personalized Plan
                    Recommendations
```

---

## User Interface

**Input (Form-based):**
- ZIP code and county
- Age(s) of covered members
- Household income (for subsidy calculation)
- Current doctors/specialists (optional)
- Current medications (optional)
- Expected healthcare usage (low/medium/high)
- Budget preference (lowest premium vs. lowest total cost)

**Output:**
- Top 3 recommended plans with:
  - Monthly premium (after subsidies)
  - Estimated annual total cost
  - Network status for their doctors
  - Coverage highlights
  - Clear "Why this plan?" explanation

---

## Data Sources

| Source | Purpose |
|--------|---------|
| Healthcare.gov API | Plan data, eligibility, subsidies |
| CMS Provider Directory | Doctor/hospital network lookup |
| RxNorm/OpenFDA | Medication coverage verification |
| RAG Vector Store | Plan details, benefits summaries, FAQ |

---

## Technical Architecture

```
Frontend (index.html)     →    FastAPI Backend    →    LangGraph Agents
                                    │
                          ┌─────────┼─────────┐
                          │         │         │
                       OpenAI    RAG Store   APIs
                        LLM     (Plan Data)  (CMS, etc.)
                          │         │         │
                          └─────────┼─────────┘
                                    │
                            Arize Observability
```

- **Backend:** FastAPI (adapted from trip planner)
- **Agent Framework:** LangGraph with parallel execution
- **LLM:** OpenAI GPT-4o-mini (cost-effective, fast)
- **RAG:** Vector search over plan documents and FAQs
- **Observability:** Arize for tracing and evaluation

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Time to recommendation | < 30 seconds |
| User completes flow | > 70% |
| Users click "View Plan" on recommendation | > 50% |
| User satisfaction (survey) | > 4.0/5.0 |
| Cost estimate accuracy | Within 15% of actual |

---

## MVP Scope (Phase 1)

- [ ] Basic form: ZIP, age, income, usage level
- [ ] Plan search via Healthcare.gov or mock data
- [ ] Cost estimation based on usage tier
- [ ] Top 3 plan recommendations with explanations
- [ ] Simple, clean UI (adapt trip planner frontend)

## Phase 2

- [ ] Doctor/provider network lookup
- [ ] Medication coverage check
- [ ] Subsidy calculator integration
- [ ] Save/compare plans feature

## Phase 3

- [ ] Chat interface for follow-up questions
- [ ] Integration with broker/carrier enrollment
- [ ] Family plan optimization
- [ ] Year-over-year plan comparison

---

## Open Questions

- [ ] Which data source for plan information? (Healthcare.gov API vs. partner data)
- [ ] How to handle state-specific exchanges (Covered California, NY State of Health)?
- [ ] Compliance requirements for providing insurance recommendations?
- [ ] Business model: affiliate fees, lead gen, or direct licensing to insurers?

---

## References

- [Healthcare.gov API Documentation](https://developer.cms.gov/)
- [AI Trip Planner Architecture](./IMPLEMENTATION_SPEC.md) (base architecture)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
