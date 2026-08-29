"""
Agent orchestrator.
Wires Planner -> Retriever -> Reasoning -> Response into a single
graph using LangGraph. This is the "AI agents that plan, retrieve,
reason, and generate" requirement from Task 8 — each node is a
separate agent with a single responsibility, and state flows
through the graph rather than being hidden in one big function.
"""

from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from app.agents import planner, retriever_agent, reasoning_agent, response_agent


class AgentState(TypedDict):
    question: str
    top_k: int
    strategy: str
    retrieved_chunks: list
    draft_answer: str
    result: dict


def _plan_node(state: AgentState) -> AgentState:
    plan = planner.plan(state["question"])
    state.update(plan)
    return state


def _retrieve_node(state: AgentState) -> AgentState:
    updated = retriever_agent.retrieve(state)
    state.update(updated)
    return state


def _reason_node(state: AgentState) -> AgentState:
    updated = reasoning_agent.reason(state)
    state.update(updated)
    return state


def _respond_node(state: AgentState) -> AgentState:
    result = response_agent.respond(state)
    state["result"] = result
    return state


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", _plan_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("reason", _reason_node)
    graph.add_node("respond", _respond_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# Compiled once at import time, reused across requests
_agent_app = None


def run_agent_query(question: str) -> dict:
    global _agent_app
    if _agent_app is None:
        _agent_app = build_agent_graph()

    final_state = _agent_app.invoke({"question": question})
    return final_state["result"]
