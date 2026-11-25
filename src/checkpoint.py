from pathlib import Path

import torch


def save_checkpoint(*, epoch, model, optim, history, best_val_loss, path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    ckpt_path = directory / "best_run.pt"

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
    }

    torch.save(state, ckpt_path)


def load_checkpoint(*, model, optim, path):
    ckpt_path = Path(path) / "best_run.pt"
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])

    return ckpt
