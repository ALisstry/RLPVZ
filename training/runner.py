from .checkpoint import CheckpointManager
from .context import build_train_context
from .evaluation import load_evaluation_config
from .game_instances import (
    prepare_eval_game_instances,
    prepare_game_instances,
    terminate_pvz_processes,
)
from .logging import setup_logging
from .paths import build_run_paths
from utils.train_utils import (
    load_training_config,
    print_metadata,
    setup_device,
    write_run_metadata,
)


class TrainRunner:
    def __init__(self, args, algorithm):
        self.args = args
        self.algorithm = algorithm

    def run(self) -> None:
        run_paths = build_run_paths(self.args)
        setup_logging(self.args, run_paths)

        checkpoint = CheckpointManager(self.args, run_paths)
        checkpoint.prepare_resume()

        print_metadata(self.args, self.algorithm, run_paths)
        device = setup_device()

        game_instances = prepare_game_instances(self.args)
        if game_instances is None:
            return
        eval_config = load_evaluation_config(
            load_training_config(self.args.training_config)
            .get("training", {})
            .get("eval", {})
        )
        eval_game_instances = prepare_eval_game_instances(self.args, eval_config)

        context = build_train_context(
            args=self.args,
            algorithm=self.algorithm,
            device=device,
            game_instances=game_instances,
            eval_game_instances=eval_game_instances,
            eval_config=eval_config,
            checkpoint=checkpoint,
            run_paths=run_paths,
        )
        write_run_metadata(context, self.algorithm)
        interrupted = False
        training_error = None
        try:
            self.algorithm.train(context)
        except KeyboardInterrupt:
            interrupted = True
            print("\r\n 训练被中断")
        except Exception as exc:
            training_error = exc
            raise
        finally:
            if training_error is not None:
                run_status = "failed"
            elif interrupted:
                run_status = "interrupted"
            else:
                run_status = "completed"
            try:
                context.checkpoint.save_payload(
                    context.artifacts.to_checkpoint_payload()
                )
            except Exception as exc:
                print(f"\n[Checkpoint] 保存失败: {exc}")
                if training_error is None and not interrupted:
                    raise
            finally:
                write_run_metadata(
                    context,
                    self.algorithm,
                    status=run_status,
                    error=training_error,
                )
                context.artifacts.close()
                terminate_pvz_processes(
                    context.game_instances,
                    auto_start=getattr(self.args, "auto_start", True),
                )
                terminate_pvz_processes(
                    context.eval_game_instances,
                    auto_start=getattr(self.args, "auto_start", True),
                )

        if interrupted:
            return
