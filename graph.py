"""LangGraph workflow engine (Module 4: Workflow Automation & Decision
Intelligence). Replaces the hand-written loop in CoordinatorAgent.handle.

    START -> plan -> run_step --(handoff)--> handoff --> run_step
                         |--(more steps)--> advance --> run_step
                         |--(action awaiting confirmation + more steps)--> pause --> finalize
                         |--(done)--> finalize -> END

Decision points
  * after every step: hand off to another agent, continue, pause, or finish;
  * pause: if a step needs the user's "yes" before it can act, the remaining
    steps are saved and resumed automatically after the confirmation, so a
    later step never runs on the assumption that an earlier one happened.

Every run records metrics (steps, handoffs, tool calls, blocked calls, time)
in session.scratchpad["workflow_runs"].
"""

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.coordinator_agent import parse_handoff
from core.logger import audit

MAX_HANDOFFS = 2
NO_TEAM_REPLY = ("I couldn't find the right HR team for that request. "
                 "Please rephrase or contact HR directly.")


class WorkflowState(TypedDict, total=False):
    query: str
    session: Any
    steps: list          # [{"agent", "task", "text"}]
    idx: int
    agent: str           # agent currently working on the step
    visited: list        # agents already tried for this step (handoff loop guard)
    hops: int
    hops_total: int      # handoffs across the whole run (for metrics)
    handoff_to: str
    replies: list
    used: list
    paused: bool
    started: float
    trace_start: int
    route: str
    reply: str
    metrics: dict


class HRWorkflow:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.agents = coordinator.agents
        self.app = self._build()

    # ---------- graph wiring ----------
    def _build(self):
        g = StateGraph(WorkflowState)
        g.add_node("plan", self._plan)
        g.add_node("run_step", self._run_step)
        g.add_node("handoff", self._handoff)
        g.add_node("advance", self._advance)
        g.add_node("pause", self._pause)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "plan")
        g.add_edge("plan", "run_step")
        g.add_conditional_edges("run_step", self._decide, {
            "handoff": "handoff", "next": "advance", "pause": "pause", "done": "finalize",
        })
        g.add_edge("handoff", "run_step")
        g.add_edge("advance", "run_step")
        g.add_edge("pause", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # ---------- nodes ----------
    def _plan(self, state):
        session, query = state["session"], state["query"]
        session.turn += 1
        confirming = self.coordinator.is_confirmation(query, session)
        raw = self.coordinator.plan(query, session)

        if confirming:                               # the confirmed step, then whatever was waiting
            steps = [{**raw[0], "text": query}]
            steps += session.scratchpad.pop("resume_steps", [])
        else:
            session.scratchpad.pop("resume_steps", None)     # stale pause from an abandoned request
            multi = len(raw) > 1
            steps = [{**s, "text": f"{s['task']}\n(Original request: {query})" if multi else query}
                     for s in raw]
        audit("workflow_plan", user=session.user_id, steps=[s["agent"] for s in steps], resumed=confirming)
        return {
            "steps": steps, "idx": 0, "agent": steps[0]["agent"], "visited": [], "hops": 0, "hops_total": 0,
            "replies": [], "used": [], "paused": False, "handoff_to": None,
            "started": time.perf_counter(),
            "trace_start": len(session.scratchpad.get("tool_trace", [])),
        }

    def _run_step(self, state):
        session, step = state["session"], state["steps"][state["idx"]]
        agent_name = state["agent"]
        reply = self.agents[agent_name].run(step["text"], session)

        target = parse_handoff(reply)
        if target is None:
            return {"replies": state["replies"] + [reply], "used": state["used"] + [agent_name],
                    "handoff_to": None}

        usable = (target in self.agents and target != agent_name
                  and target not in state["visited"] and state["hops"] < MAX_HANDOFFS)
        if usable:
            return {"handoff_to": target}
        return {"replies": state["replies"] + [NO_TEAM_REPLY], "used": state["used"] + [agent_name],
                "handoff_to": None}

    def _handoff(self, state):
        session, step = state["session"], state["steps"][state["idx"]]
        source, target = state["agent"], state["handoff_to"]
        audit("workflow_handoff", source=source, target=target)
        session.post(source, target, f"Request handed over to you: {step['text']}")
        return {"agent": target, "visited": state["visited"] + [source],
                "hops": state["hops"] + 1, "hops_total": state["hops_total"] + 1,
                "handoff_to": None}

    def _advance(self, state):
        session, steps, idx = state["session"], state["steps"], state["idx"]
        nxt = steps[idx + 1]
        session.post(state["used"][-1], nxt["agent"],
                     f"I completed '{steps[idx]['task']}'. My result: {state['replies'][-1]}")
        return {"idx": idx + 1, "agent": nxt["agent"], "visited": [], "hops": 0}

    def _pause(self, state):
        session = state["session"]
        remaining = state["steps"][state["idx"] + 1:]
        session.scratchpad["resume_steps"] = remaining
        audit("workflow_paused", user=session.user_id, waiting=[s["agent"] for s in remaining])
        return {"paused": True}

    def _finalize(self, state):
        session = state["session"]
        used, replies = state["used"], state["replies"]
        if len(replies) == 1:
            route, reply = used[0], replies[0]
        else:
            route = "+".join(used)
            reply = "\n\n".join(f"[{a}] {r}" for a, r in zip(used, replies))
        if state.get("paused"):
            left = session.scratchpad.get("resume_steps", [])
            reply += (f"\n\n(Paused: {len(left)} more step(s) will continue "
                      "once you confirm the action above.)")

        trace = session.scratchpad.get("tool_trace", [])[state["trace_start"]:]
        metrics = {
            "steps_run": len(replies),
            "agents": used,
            "handoffs": state["hops_total"],
            "tool_calls": len(trace),
            "blocked_calls": sum(1 for t in trace if t["outcome"] == "blocked"),
            "paused": bool(state.get("paused")),
            "elapsed_ms": round((time.perf_counter() - state["started"]) * 1000, 1),
        }
        session.scratchpad.setdefault("workflow_runs", []).append(metrics)
        audit("workflow_done", user=session.user_id, route=route, **metrics)
        return {"route": route, "reply": reply, "metrics": metrics}

    # ---------- decision point ----------
    def _decide(self, state):
        if state.get("handoff_to"):
            return "handoff"
        session = state["session"]
        has_more = state["idx"] + 1 < len(state["steps"])
        if not has_more:
            return "done"
        pending = session.pending_action
        if pending and pending["turn"] == session.turn:      # a step is waiting for the user's yes
            return "pause"
        return "next"

    # ---------- public API ----------
    def run(self, query: str, session):
        """Same contract as CoordinatorAgent.handle: returns (route, reply)."""
        final = self.app.invoke({"query": query, "session": session})
        return final["route"], final["reply"]

    def mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()
