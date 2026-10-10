"""Tool 4 - HR analytics (Outcome: "Intelligent Decision Support" -- analyse
information, evaluate options, recommend actions).

Computes AGGREGATE statistics from the employee and leave data and attaches
rule-based recommendations. It never returns individual names or IDs, so it
is safe to use without per-user authorisation.
"""

from collections import Counter

from langchain_core.tools import tool

from tools.mock_data import EMPLOYEES, LEAVE_BALANCES

# Yearly entitlement, matching the "leave policy" text in mock_data.POLICIES.
ENTITLEMENT = {"annual": 12, "sick": 6, "casual": 4}
EXIT_STATUSES = ("exiting", "access_revoked")
METRICS = ("headcount", "leave_usage", "attrition", "summary")

LOW_BALANCE_DAYS = 1          # "nearly out of annual leave"
HIGH_USAGE_PERCENT = 75       # "using most of an entitlement"
CONCENTRATION_PERCENT = 50    # share of a department that is leaving


def _headcount() -> tuple:
    by_status = Counter(e["status"] for e in EMPLOYEES.values())
    by_dept = Counter(e["department"] for e in EMPLOYEES.values())
    text = (f"Headcount: {len(EMPLOYEES)} employee records. "
            f"By status: {', '.join(f'{k} {v}' for k, v in sorted(by_status.items()))}. "
            f"By department: {', '.join(f'{k} {v}' for k, v in sorted(by_dept.items()))}.")
    return text, []


def _leave_usage() -> tuple:
    rows = [b for emp_id, b in LEAVE_BALANCES.items() if emp_id in EMPLOYEES]
    if not rows:
        return "Leave usage: no leave data available.", []
    parts, advice = [], []
    for leave_type, total in ENTITLEMENT.items():
        used = [max(total - b.get(leave_type, total), 0) for b in rows]
        percent = round(100 * sum(used) / (total * len(rows)))
        parts.append(f"{leave_type} {percent}% of entitlement used (avg {sum(used) / len(rows):.1f} of {total} days)")
        if percent >= HIGH_USAGE_PERCENT:
            advice.append(f"{leave_type.capitalize()} leave is {percent}% used on average; "
                          "plan cover for the rest of the year.")
    low = sum(1 for b in rows if b.get("annual", 99) <= LOW_BALANCE_DAYS)
    if low:
        advice.append(f"{low} employee(s) have {LOW_BALANCE_DAYS} or fewer annual leave days left; "
                      "check upcoming workload before approving more leave.")
    return "Leave usage: " + "; ".join(parts) + ".", advice


def _attrition() -> tuple:
    total = len(EMPLOYEES)
    leaving = [e for e in EMPLOYEES.values() if e["status"] in EXIT_STATUSES]
    if not total:
        return "Attrition: no employee data available.", []
    rate = round(100 * len(leaving) / total, 1)
    advice = []
    exiting_active = sum(1 for e in leaving if e["status"] == "exiting")
    if exiting_active:
        advice.append(f"{exiting_active} exiting employee(s) still have active access; "
                      "schedule access revocation for their last working day.")
    dept_total = Counter(e["department"] for e in EMPLOYEES.values())
    for dept, n_leaving in Counter(e["department"] for e in leaving).items():
        share = round(100 * n_leaving / dept_total[dept])
        if share >= CONCENTRATION_PERCENT:
            advice.append(f"{dept} is losing {n_leaving} of {dept_total[dept]} people ({share}%); "
                          "start backfill and handover planning.")
    return f"Attrition: {len(leaving)} of {total} employees are in the exit process ({rate}%).", advice


@tool
def get_hr_analytics(metric: str = "summary") -> str:
    """Get aggregate HR analytics with recommendations. metric is one of
    'headcount', 'leave_usage', 'attrition' or 'summary' (all three). Returns totals and
    percentages only, never individual employees."""
    metric = (metric or "summary").strip().lower().replace(" ", "_")
    if metric not in METRICS:
        return f"Unknown metric '{metric}'. Choose one of: {', '.join(METRICS)}."
    if not EMPLOYEES:
        return "No employee data is available to analyse."

    builders = {"headcount": [_headcount], "leave_usage": [_leave_usage], "attrition": [_attrition],
                "summary": [_headcount, _leave_usage, _attrition]}[metric]
    facts, advice = [], []
    for build in builders:
        text, tips = build()
        facts.append(text)
        advice += tips
    out = "\n".join(facts)
    out += ("\nRecommendations:\n" + "\n".join(f"- {a}" for a in advice)) if advice \
        else "\nRecommendations: none; no issues detected."
    return out
