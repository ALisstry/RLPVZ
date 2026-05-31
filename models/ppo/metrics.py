from training.metrics import TrainingCurveWriter


def build_metrics_writers(args, run_paths):
    if getattr(args, "ppo_plot_freq", 0) <= 0:
        return []

    return [
        TrainingCurveWriter(
            output_path=getattr(args, "ppo_plot_path", None)
            or run_paths.training_curve_path,
            refresh_freq=getattr(args, "ppo_plot_freq", 20),
        )
    ]
