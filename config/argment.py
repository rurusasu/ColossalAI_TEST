from argparse import ArgumentParser


def My_parser():
    parser = ArgumentParser(description="Trainer for AlexNet", allow_abbrev=False)

    # Standard arguments.
    parser.add_argument("--config", type=str, default="./config/config.py")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="the master address for distributed training",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12345,
        help="the master port for distributed training",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="world size for distributed training"
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="rank for the default process parser"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank on the node"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["nccl", "gloo"],
        default="nccl",
        help="backend for distributed communication",
    )

    return parser


if __name__ == "__main__":
    args = parse_args()
    print(args)
