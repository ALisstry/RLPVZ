class DDQNConsoleReporter:
    def print_progress(self, episode_stats):
        print("\r" + episode_stats.progress_line, end="", flush=True)

    def print_eval(self, eval_stats, progress_line):
        print(
            f"\n[Eval] Episode {eval_stats.episode} | "
            f"avg_score={eval_stats.avg_score:.2f} | "
            f"avg_iter={eval_stats.avg_iterations:.2f}",
            flush=True,
        )
        print("\r" + progress_line, end="", flush=True)

    def print_checkpoint(self, episode_count):
        print(
            f"\n[DDQN] 已保存周期 checkpoint: episode {episode_count}",
            flush=True,
        )

    def print_finished(self, solved, episode_count):
        if solved:
            print(f"\nEnvironment solved in {episode_count} episodes.")
        else:
            print("\nEpisode limit reached.")
