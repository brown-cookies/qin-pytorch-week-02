import sys

import torch

from src.cli import parse_args
from src.config import get_config
from src.data.pipeline import prepare_data
from src.loader import make_loader
from src.model import ReadmittanceModel
from src.split import split_train_val_test
from src.train import train_model


def run():
    cfg = get_config()

    torch.manual_seed(cfg.seed)

    df = prepare_data()

    d = split_train_val_test(df=df, cfg=cfg)
    X_train = d["X_train"]
    X_val = d["X_val"]
    X_test = d["X_test"]
    y_train = d["y_train"]
    y_val = d["y_val"]
    y_test = d["y_test"]

    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=cfg.batch_size,
        shuffle=True
    )
    val_loader = make_loader(
        X_val,
        y_val,
        batch_size=cfg.batch_size
    )
    test_loader = make_loader(
        X_test,
        y_test,
        cfg.batch_size
    )

    n_features = X_train.shape[1]

    model = ReadmittanceModel(n_features).to(cfg.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        cfg=cfg
    )

    print(history)


def command_train(out, is_smoke=False):
    ...


def command_evaluate(checkpoint, out):
    ...


def command_split(out):
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
