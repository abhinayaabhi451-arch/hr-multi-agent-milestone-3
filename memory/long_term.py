"""Long-term knowledge store (Milestone 3: Agent Coordination & Memory
Management). Persists facts about each employee across runs.
"""

import json
import os
from threading import Lock

_STORE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_store.json")
_lock = Lock()


def _load() -> dict:
    if not os.path.exists(_STORE_PATH):
        return {}
    with open(_STORE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save(data: dict) -> None:
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_fact(employee_id: str, key: str, value: str) -> None:
    with _lock:
        data = _load()
        data.setdefault(employee_id, {})[key] = value
        _save(data)


def get_fact(employee_id: str, key: str, default=None):
    with _lock:
        data = _load()
        return data.get(employee_id, {}).get(key, default)


def get_all_facts(employee_id: str) -> dict:
    with _lock:
        data = _load()
        return data.get(employee_id, {})