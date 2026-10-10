"""Tool 3 - Knowledge-base search (Module 2: "external knowledge sources").

Searches company documents (handbook, benefits guide, IT policy ...) and
returns the best-matching passages, so agents answer from real text instead
of guessing. It reads two sources:

  1. SAMPLE_DOCUMENTS below (built in, so the tool works out of the box);
  2. any .txt / .md files you drop into the  knowledge_base/  folder next to
     the project -- paragraphs separated by a blank line. Replace the sample
     content with your company's real documents there.
"""

import math
import os
import re

from langchain_core.tools import tool

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
MAX_QUERY_CHARS = 200
MAX_PASSAGES = 2
MAX_PASSAGE_CHARS = 600

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "do", "does", "can", "i", "my", "me", "we", "our", "you", "your",
    "what", "how", "many", "much", "when", "where", "who", "which", "to", "of", "for", "in", "on", "at",
    "and", "or", "it", "this", "that", "about", "tell", "please", "get", "have", "has", "be",
}

SAMPLE_DOCUMENTS = [
    {"title": "Employee Handbook - Code of Conduct",
     "text": "All employees are expected to act with honesty and respect. Harassment or discrimination of any kind "
             "is not tolerated and should be reported to HR or through the anonymous ethics line.\n\n"
             "Conflicts of interest must be disclosed to your manager. Gifts from vendors worth more than "
             "INR 2,000 must be declined or reported to HR."},
    {"title": "Benefits Guide - Health and Insurance",
     "text": "Every permanent employee is covered by group health insurance from the first day of joining. "
             "Cover extends to the employee, spouse and up to two children.\n\n"
             "Employees also receive accidental death cover and are enrolled in the provident fund (PF) "
             "after joining; the employer and employee each contribute 12% of basic pay."},
    {"title": "Expense Reimbursement Policy",
     "text": "Business expenses must be submitted within 30 days with original receipts. Travel and meal "
             "claims above INR 5,000 need prior manager approval.\n\n"
             "Reimbursements are paid together with the next monthly payroll after approval."},
    {"title": "IT and Security Policy",
     "text": "Company laptops must use full-disk encryption and a screen lock of five minutes or less. "
             "Passwords must be changed every 90 days and multi-factor authentication is mandatory.\n\n"
             "Report lost devices or suspected phishing to the IT helpdesk immediately."},
    {"title": "Exit and Notice Period Guidelines",
     "text": "Employees must serve a notice period of 60 days after submitting their resignation, unless HR "
             "agrees to an early release. An exit interview is held in the final two weeks.\n\n"
             "Company assets such as the laptop and ID badge must be returned on or before the last working day."},
]


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _tokens(text: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOPWORDS}


def _load_folder_documents() -> list:
    docs = []
    if not os.path.isdir(KB_DIR):
        return docs
    for name in sorted(os.listdir(KB_DIR)):
        if name.lower().endswith((".txt", ".md")):
            try:
                with open(os.path.join(KB_DIR, name), "r", encoding="utf-8") as f:
                    docs.append({"title": os.path.splitext(name)[0].replace("_", " ").title(), "text": f.read()})
            except OSError:
                continue                       # unreadable file: skip it, don't fail the whole search
    return docs


def _passages() -> list:
    out = []
    for doc in SAMPLE_DOCUMENTS + _load_folder_documents():
        for para in re.split(r"\n\s*\n", doc["text"]):
            para = para.strip()
            if para:
                out.append((doc["title"], para))
    return out


@tool
def search_knowledge_base(query: str) -> str:
    """Search company documents (handbook, benefits, expenses, IT security, exit guidelines)
    and return the most relevant passages. Use for questions the HR policy lookup can't answer,
    e.g. 'notice period', 'health insurance cover', 'expense claim deadline'."""
    query = (query or "").strip()
    if not query:
        return "Please tell me what to search the knowledge base for."
    if len(query) > MAX_QUERY_CHARS:
        return f"The search text is too long ({len(query)} characters; maximum {MAX_QUERY_CHARS}). Please shorten it."
    wanted = _tokens(query)
    if not wanted:
        return "Please use more specific words to search the knowledge base."

    needed = max(1, math.ceil(len(wanted) / 2))        # at least half of the meaningful words must match
    scored = []
    for title, para in _passages():
        score = len(wanted & _tokens(para)) + 0.5 * len(wanted & _tokens(title))
        if score >= needed:
            scored.append((score, title, para))
    if not scored:
        return f"No knowledge-base document matches '{query}'."

    scored.sort(key=lambda s: -s[0])
    lines = []
    for i, (_, title, para) in enumerate(scored[:MAX_PASSAGES], 1):
        text = para if len(para) <= MAX_PASSAGE_CHARS else para[:MAX_PASSAGE_CHARS].rsplit(" ", 1)[0] + " ..."
        lines.append(f"[{i}] {title}: {text}")
    return "Knowledge-base results (answer only from these passages):\n" + "\n".join(lines)
