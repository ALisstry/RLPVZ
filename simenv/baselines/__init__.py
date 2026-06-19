"""
Baseline agents for SimPVZ evaluation.

Each agent exposes a uniform interface:
    agent.select_action(env: SimPVZEnv) -> int
    agent.name -> str
"""

from .random_agent import RandomAgent
from .heuristic_agent import SimpleHeuristicAgent, SmartHeuristicAgent

__all__ = ["RandomAgent", "SimpleHeuristicAgent", "SmartHeuristicAgent"]
