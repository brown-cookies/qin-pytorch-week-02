import json
import sys

from src.cli import parse_args


def run():
    ...


def train(out, is_smoke=False):
    ...


def evaluate(checkpoint, out):
    ...


def split(out):
    ...


def main(argv=None):
    """Entry point of the project."""

    args = parse_args(argv)

    if args.command == "run":
        run()
    elif args.command == "train":
        train(args.out, args.smoke)
    elif args.command == "evaluate":
        evaluate(args.checkpoint, args.metrics_out)
    elif args.command == "split":
        split(args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
