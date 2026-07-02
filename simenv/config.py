DIFFICULTY = {
    "fps": 2,
    "max_frames": 650,
    "rows": 5,
    "cols": 9,
    "initial_sun": 50,
    "natural_sun_production": 25,
    "natural_sun_cooldown": 5,
    "mowers": False,
    "survival_score": 0,
    "survival_step": 20,
    "alive_plant_score": 0,
    "alive_mower_score": 0,
}


ZOMBIE_SPAWN = {
    "first_wave_delay_sec": 18,
    "wave_interval_sec": 27,
    "post_clear_wave_delay_sec": 3,
    "flag_wave_followup_interval_sec": 50,
    "flag_wave_modulo": 10,
    "flag_wave_remainder": 9,
    "base_advanced_probability": 0.05,
    "advanced_probability_growth": 2.0,
    "max_advanced_probability": 1.0,
    "cone_probability_multiplier": 3.0,
}


REWARDS = {
    "zombie_kill": {
        "use_type_rewards": True,
        "default": 8.00,
        "zombie": 8.00,
        "flag": 12.00,
        "conehead": 12.00,
        "buckethead": 20.00,
    },
    "sun_collect": 0.01,
    "wave_complete": 20.0,
    "game_win": 50.0,
    "streak_bonus": 0.0,
    "survival_per_step": 0.0,
    "plant_sunflower": 0.10,
    "plant_attacker": 0.35,
    "plant_wall": 0.18,
    "plant_other": 0.30,
    "plant_lost": -0.25,
    "sunflower_lost": -0.80,
    "lawnmower_triggered": -50.0,
    "game_lose": -6.0,
    "invalid_action": -0.01,
    "wait_with_sun": -0.02,
    "wait_sun_threshold": 300,
    "coverage": {
        "scale": 0.0,
    },
    "proximity": {
        "scale": 0.0,
    },
    "potential": {
        "sun_scale": 0.06,
        "sun_cap": 300.0,
        "plant_scale": 0.22,
        "spread_bonus": 0.06,
        "lawnmower_scale": 0.35,
        "zombie_threat_scale": 0.35,
        "zombie_distance_bonus": 0.75,
        "wave_scale": 0.05,
        "delta_scale": 0.18,
    },
}
