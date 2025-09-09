from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Minimal tools (deterministic for tutorials)
@tool
def essential_info(destination: str) -> str:
    """Return essential destination info like weather, sights, and etiquette."""
    return f"Key info for {destination}: mild weather, popular sights, local etiquette."


@tool
def budget_basics(destination: str, duration: str) -> str:
    """Return high-level budget categories for a given destination and duration."""
    return f"Budget for {destination} over {duration}: lodging, food, transit, attractions."


@tool
def local_flavor(destination: str, interests: Optional[str] = None) -> str:
    """Suggest authentic local experiences matching optional interests."""
    return f"Local experiences for {destination}: authentic food, culture, and {interests or 'top picks'}."


@tool
def day_plan(destination: str, day: int) -> str:
    """Return a simple day plan outline for a specific day number."""
    return f"Day {day} in {destination}: breakfast, highlight visit, lunch, afternoon walk, dinner."


# Additional simple tools per agent (to mirror original multi-tool behavior)
@tool
def weather_brief(destination: str) -> str:
    """Return a brief weather summary for planning purposes."""
    return f"Weather in {destination}: generally mild; check season for specifics."


@tool
def visa_brief(destination: str) -> str:
    """Return a brief visa guidance placeholder for tutorial purposes."""
    return f"Visa guidance for {destination}: check your nationality's embassy site."


@tool
def attraction_prices(destination: str, attractions: Optional[List[str]] = None) -> str:
    """Return rough placeholder prices for attractions."""
    items = attractions or ["Museum", "Historic Site", "Viewpoint"]
    priced = ", ".join(f"{a}: $10-$40" for a in items)
    return f"Attraction prices in {destination}: {priced}"


@tool
def local_customs(destination: str) -> str:
    """Return simple etiquette reminders for the destination."""
    return f"Customs in {destination}: be polite, modest dress in sacred places, learn greetings."


@tool
def hidden_gems(destination: str) -> str:
    """Return a few off-the-beaten-path ideas."""
    return f"Hidden gems in {destination}: small cafes, local markets, lesser-known viewpoints."


@tool
def travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Return an approximate travel time placeholder."""
    return f"Travel from {from_location} to {to_location} by {mode}: ~20-60 minutes."


@tool
def packing_list(destination: str, duration: str, activities: Optional[List[str]] = None) -> str:
    """Return a generic packing list summary."""
    acts = ", ".join(activities or ["walking", "sightseeing"]) 
    return f"Packing for {destination} ({duration}): comfortable shoes, layers, adapter; for {acts}."


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research: Optional[str]
    budget: Optional[str]
    local: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def research_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    prompt_t = (
        "You are a research assistant.\n"
        "Gather essential information about {destination}.\n"
        "Use at most one tool if needed."
    )
    vars_ = {"destination": destination}
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        agent = llm.bind_tools([essential_info, weather_brief, visa_brief])
        res = agent.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    out = res.content
    calls: List[Dict[str, Any]] = []
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode([essential_info, weather_brief, visa_brief])
        tr = tool_node.invoke({"messages": [res]})
        msgs = tr["messages"]
        out = msgs[-1].content if msgs else out

    return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def budget_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination, duration = req["destination"], req["duration"]
    prompt_t = (
        "You are a budget analyst.\n"
        "Summarize high-level costs for {destination} over {duration}."
    )
    vars_ = {"destination": destination, "duration": duration}
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        agent = llm.bind_tools([budget_basics, attraction_prices])
        res = agent.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    out = res.content
    calls: List[Dict[str, Any]] = []
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode([budget_basics, attraction_prices])
        tr = tool_node.invoke({"messages": [res]})
        msgs = tr["messages"]
        out = msgs[-1].content if msgs else out

    return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def local_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    interests = req.get("interests", "local culture")
    prompt_t = (
        "You are a local guide.\n"
        "Suggest authentic experiences in {destination} for interests: {interests}."
    )
    vars_ = {"destination": destination, "interests": interests}
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        agent = llm.bind_tools([local_flavor, local_customs, hidden_gems])
        res = agent.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    out = res.content
    calls: List[Dict[str, Any]] = []
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "local", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode([local_flavor, local_customs, hidden_gems])
        tr = tool_node.invoke({"messages": [res]})
        msgs = tr["messages"]
        out = msgs[-1].content if msgs else out

    return {"messages": [SystemMessage(content=out)], "local": out, "tool_calls": calls}


def itinerary_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    duration = req["duration"]
    travel_style = req.get("travel_style", "standard")
    prompt_t = (
        "Create a {duration} itinerary for {destination} ({travel_style}).\n\n"
        "Inputs:\nResearch: {research}\nBudget: {budget}\nLocal: {local}\n"
    )
    vars_ = {
        "duration": duration,
        "destination": destination,
        "travel_style": travel_style,
        "research": (state.get("research") or "")[:400],
        "budget": (state.get("budget") or "")[:400],
        "local": (state.get("local") or "")[:400],
    }
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_graph():
    g = StateGraph(TripState)
    g.add_node("research_node", research_agent)
    g.add_node("budget_node", budget_agent)
    g.add_node("local_node", local_agent)
    g.add_node("itinerary_node", itinerary_agent)

    g.add_edge(START, "research_node")
    g.add_edge("research_node", "budget_node")
    g.add_edge("budget_node", "local_node")
    g.add_edge("local_node", "itinerary_node")
    g.add_edge("itinerary_node", END)

    return g.compile(checkpointer=MemorySaver())


app = FastAPI(title="AI Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-trip-planner"}


@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):
    if _TRACING:
        try:
            space_id = os.getenv("ARIZE_SPACE_ID")
            api_key = os.getenv("ARIZE_API_KEY")
            if space_id and api_key:
                tp = register(space_id=space_id, api_key=api_key, project_name="ai-trip-planner")
                LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
                LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
        except Exception:
            pass

    graph = build_graph()
    state = {
        "messages": [],
        "trip_request": req.model_dump(),
        "research": None,
        "budget": None,
        "local": None,
        "final": None,
        "tool_calls": [],
    }
    cfg = {"configurable": {"thread_id": f"tut_{req.destination}_{datetime.now().strftime('%H%M%S')}"}}
    out = graph.invoke(state, cfg)
    return TripResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
