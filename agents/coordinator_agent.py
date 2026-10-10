"""Coordinator Agent (Milestone 3: Agent Coordination & Memory
Management). Routes an incoming message to the right specialized agent.
"""

from langchain_core.prompts import ChatPromptTemplate
from core.llm_client import get_llm, extract_text

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing classifier for an HR multi-agent system.
Given an employee's message, decide which single agent should handle it.
Respond with ONLY one word, exactly one of:
chatbot, onboarding, offboarding, maintenance

Routing guide:
- chatbot: general questions, FAQs, company policy questions not tied to a specific process
- onboarding: new hire document submission, training schedules, joining steps
- offboarding: resignation, exit interviews, access revocation, final settlement
- maintenance: leave requests, payroll queries, performance updates

Respond with only the one word, nothing else."""),
    ("human", "{query}")
])


class CoordinatorAgent:
    VALID_ROUTES = ("chatbot", "onboarding", "offboarding", "maintenance")

    def __init__(self, agents: dict):
        self.llm = get_llm()
        self.chain = ROUTER_PROMPT | self.llm
        self.agents = agents

    def route(self, query: str) -> str:
        response = self.chain.invoke({"query": query})
        decision = extract_text(response.content).strip().lower()
        for valid in self.VALID_ROUTES:
            if valid in decision:
                return valid
        return "chatbot"

    def handle(self, query: str):
        route = self.route(query)
        agent = self.agents[route]
        return route, agent.run(query)