import os

from callbacks import (
    AutoCollectCallback,
    AsyncSingleModelCallback,
    DetailedLogCallback,
    DynamicEntropyCallback,
    HeatmapCallback,
    MemoryResetCallback,
    PPOMetricsCallback,
    SimpleMonitorCallback,
)


def build_callbacks(args, run_paths, checkpoint=None, metrics=None):
    os.makedirs(run_paths.output_dir, exist_ok=True)

    callbacks = [
        MemoryResetCallback(verbose=0),
        AutoCollectCallback(),
        SimpleMonitorCallback(),
        AsyncSingleModelCallback(
            save_freq=args.save_freq,
            save_path=run_paths.cached_model_path,
            checkpoint=checkpoint,
            verbose=1,
        ),
        HeatmapCallback(
            save_path=run_paths.heatmap_path,
            refresh_rate=10,
            verbose=1,
        ),
        DetailedLogCallback(log_freq=500),
        PPOMetricsCallback(metrics=metrics),
    ]

    callbacks.append(
        DynamicEntropyCallback(
            start_ent_coef=args.start_ent,
            end_ent_coef=args.end_ent,
            decay_type=args.ent_decay,
            total_timesteps=args.timesteps,
            warmup_steps=min(10000, args.timesteps // 10),
            verbose=0,
        )
    )
    return callbacks
