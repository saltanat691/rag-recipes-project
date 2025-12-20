from dataclasses import dataclass
from typing import List, Dict

@dataclass
class WeeklyPlan:
    days: List[Dict]          # [{"day": "Mon", "breakfast": "...", ...}, ...]
    grocery_list: Dict[str, List[str]]  # {"produce": [...], "meat": [...], ...}

class Planner:
    """
    Placeholder: will later compose a 7-day plan from retrieved recipes + constraints.
    Keep it here so the architecture is ready without touching the RAG core.
    """
    def create_weekly_plan(self, constraints: dict) -> WeeklyPlan:
        raise NotImplementedError("Planner mode not implemented yet.")
