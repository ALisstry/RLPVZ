"""
Simulation environment training entry point.

Usage: python train_sim.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simenv.trainer import train_sim


if __name__ == "__main__":
    train_sim(
        max_episodes=60000,
        buffer_size=60000,
        burn_in=10000,
        batch_size=200,
    )
