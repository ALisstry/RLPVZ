from training.args import get_args
import training.constants as const
from training.runner import TrainRunner
from training.registry import create_algorithm
from utils.train_utils import set_torch_num_threads

def main():
    set_torch_num_threads(const.TORCH_THREADS_NUM)

    args = get_args()
    algorithm = create_algorithm(args.algo, args)
    runner = TrainRunner(args, algorithm)
    runner.run()


if __name__ == "__main__":
    main()
