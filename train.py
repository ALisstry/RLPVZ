from train_args import get_args
from training.bootstrap import configure_training_process


def main():
    configure_training_process()
    from training import TrainRunner, create_algorithm

    args = get_args()
    algorithm = create_algorithm(args.algo, args)
    runner = TrainRunner(args, algorithm)
    runner.run()


if __name__ == "__main__":
    main()

