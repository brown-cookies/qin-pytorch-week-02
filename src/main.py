import json
from pathlib import Path
import sys

import torch

from src.baseline import evaluate_baseline, get_gradient_boosted_baseline, get_trivial_baseline
from src.cli import parse_args
from src.config import get_config
from src.data.features import PATIENT_ID
from src.data.pipeline import prepare_data
from src.eval import evaluate_model
from src.helper import print_numbers
from src.loader import make_loader
from src.model import ReadmittanceModel
from src.split import split_indices, split_train_val_test
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

    trivial_baseline = get_trivial_baseline(X_train, y_train)
    gradient_boosted_baseline = get_gradient_boosted_baseline(
        X_train, y_train, cfg=cfg
    )

    model = ReadmittanceModel(n_features).to(cfg.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=d["pos_weight"])

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        resume=False,
        checkpoint_path="runs/",
        cfg=cfg
    )

    results = {
        "trivial": evaluate_baseline(trivial_baseline, X_val, y_val, threshold=cfg.threshold),
        "hgb": evaluate_baseline(gradient_boosted_baseline, X_val, y_val, threshold=cfg.threshold),
        "model": evaluate_model(model=model, loader=val_loader, threshold=cfg.threshold)
    }

    print_numbers(results)


def command_train(out, is_smoke=False):
    ...


def command_evaluate(checkpoint, out):
    ...


def command_split(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = get_config()

    torch.manual_seed(cfg.seed)

    df = prepare_data()

    indices = split_indices(
        groups=df[PATIENT_ID].to_numpy(),
        val_split=cfg.val_size_split,
        test_split=cfg.test_size_split,
        seed=cfg.seed
    )

    split_data = {
        key: value.tolist()
        for key, value in indices.items()
    }

    out.write_text(
        json.dumps(split_data, indent=4),
        encoding="utf-8"
    )


def main(argv=None):
    """Entry point of the project."""

    args = parse_args(argv)

    if args.command == "run":
        run()
    elif args.command == "train":
        command_train(args.out, args.smoke)
    elif args.command == "evaluate":
        command_evaluate(args.checkpoint, args.metrics_out)
    elif args.command == "split":
        command_split(args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
