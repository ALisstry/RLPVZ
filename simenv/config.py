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
    "use_shaped": False,
    "reward_scale": 1.0,
    "plant_lost_sun_cost_scale": 0.0,
    "zombie_kill": {
        "use_type_rewards": True,
        "default": 8.00,
        "zombie": 8.00,
        "flag": 12.00,
        "conehead": 12.00,
        "buckethead": 20.00,
    },
    "sun_collect": 0.0,
    "wave_complete": 0.0,
    "game_win": 0.0,
    "streak_bonus": 0.0,
    "survival_per_step": 0.0,
    "plant_sunflower": 0.0,
    "plant_attacker": 0.0,
    "plant_wall": 0.0,
    "plant_other": 0.0,
    "plant_lost": 0.0,
    "sunflower_lost": 0.0,
    "lawnmower_triggered": 0.0,
    "game_lose": 0.0,
    "invalid_action": 0.0,
    "wait_with_sun": 0.0,
    "wait_sun_threshold": 300,
    "coverage": {
        "scale": 0.0,
    },
    "proximity": {
        "scale": 0.0,
    },
    "potential": {
        "sun_scale": 0.0,
        "sun_cap": 300.0,
        "plant_scale": 0.0,
        "spread_bonus": 0.0,
        "lawnmower_scale": 0.0,
        "zombie_threat_scale": 0.0,
        "zombie_distance_bonus": 0.0,
        "wave_scale": 0.0,
        "delta_scale": 0.0,
    },
}

from simenv.consts import Plants

CURRICULUM = {
    "enabled": True,
    "strategy": "stage_gate",
    "default_burn_in": None,
    "default_stage_epsilon": {
        "start": 0.35,
        "end": 0.05,
        "decay_episodes": 20000,
    },
    "stage_gate": {
        "stages": [
            {
                "stage_name": "sim_melon_economy_bootstrap_3rows",
                "enabled_rows": [0, 1, 2],
                "enabled_plants": [
                    Plants.Sunflower,
                    Plants.Melon_pult,
                ],  # Teach melon as the first long-term attacker with buffered economy.
                "initial_sun": 300,
                "target_frames": 420,
                "target_flag_waves": 0,
                "timeout_frames": 560,
                "min_episodes": 5000,
                "mean_reward_threshold": 100.0,
                "mean_success_rate_threshold": 0.80,
                "epsilon": {
                    "start": 0.80,
                    "end": 0.08,
                    "decay_episodes": 7500,
                },
            },
            {
                "stage_name": "sim_wallnut_melon_3rows",
                "enabled_rows": [0, 1, 2],
                "enabled_plants": [
                    Plants.Sunflower,
                    Plants.Wallnut,
                    Plants.Melon_pult,
                ],  # Add wall-nut protection before adding instant stall tools.
                "initial_sun": 250,
                "target_frames": 520,
                "target_flag_waves": 1,
                "timeout_frames": 780,
                "min_episodes": 5000,
                "mean_reward_threshold": 140.0,
                "mean_success_rate_threshold": 0.80,
                "epsilon": {
                    "start": 0.70,
                    "end": 0.08,
                    "decay_episodes": 7500,
                },
            },
            {
                "stage_name": "sim_stall_to_melon_3rows",
                "enabled_rows": [0, 1, 2],
                "enabled_plants": [
                    Plants.Sunflower,
                    Plants.Potato_Mine,
                    Plants.Wallnut,
                    Plants.Squash,
                    Plants.Melon_pult,
                ],  # Add potato mine and squash to bridge into melon setup.
                "initial_sun": 150,
                "target_frames": 600,
                "target_flag_waves": 1,
                "timeout_frames": 900,
                "min_episodes": 7500,
                "mean_reward_threshold": 170.0,
                "mean_success_rate_threshold": 0.80,
                "epsilon": {
                    "start": 0.70,
                    "end": 0.08,
                    "decay_episodes": 7500,
                },
            },
            {
                "stage_name": "sim_cherry_stall_melon_full_rows",
                "enabled_rows": [0, 1, 2, 3, 4],
                "enabled_plants": [
                    Plants.Sunflower,
                    Plants.Potato_Mine,
                    Plants.Wallnut,
                    Plants.Squash,
                    Plants.Cherry_Bomb,
                    Plants.Melon_pult,
                ],  # Add cherry bomb as full-row emergency support.
                "initial_sun": 100,
                "target_frames": 700,
                "target_flag_waves": 2,
                "timeout_frames": 1200,
                "min_episodes": 10000,
                "mean_reward_threshold": 210.0,
                "mean_success_rate_threshold": 0.75,
                "epsilon": {
                    "start": 0.65,
                    "end": 0.08,
                    "decay_episodes": 10000,
                },
            },
            {
                "stage_name": "sim_cherry_melon_core_real_sun",
                "enabled_rows": [0, 1, 2, 3, 4],
                "enabled_plants": [
                    Plants.Sunflower,
                    Plants.Potato_Mine,
                    Plants.Wallnut,
                    Plants.Squash,
                    Plants.Cherry_Bomb,
                    Plants.Melon_pult,
                ],  # Lower to real-like opening sun while keeping cherry as emergency support.
                "initial_sun": 50,
                "target_frames": 700,
                "target_flag_waves": 2,
                "timeout_frames": 1200,
                "min_episodes": 12500,
                "mean_reward_threshold": 210.0,
                "mean_success_rate_threshold": 0.75,
                "epsilon": {
                    "start": 0.55,
                    "end": 0.06,
                    "decay_episodes": 15000,
                },
            },
            {
                "stage_name": "sim_cherry_melon_core_three_flags",
                "enabled_rows": [0, 1, 2, 3, 4],
                "enabled_plants": [
                    Plants.Sunflower,
                    Plants.Potato_Mine,
                    Plants.Wallnut,
                    Plants.Squash,
                    Plants.Cherry_Bomb,
                    Plants.Melon_pult,
                ],  # Final target extends the cherry-supported melon setup to three flag waves.
                "initial_sun": 50,
                "target_frames": 800,
                "target_flag_waves": 3,
                "timeout_frames": 1500,
                "min_episodes": 10000,
                "mean_reward_threshold": 250.0,
                "mean_success_rate_threshold": 0.65,
                "epsilon": {
                    "start": 0.35,
                    "end": 0.05,
                    "decay_episodes": 20000,
                },
            },
        ],
    },
}
