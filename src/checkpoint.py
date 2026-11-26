from pathlib import Path

import torch


def save_checkpoint(*, epoch, model, optim, history, best_val_loss, path, name="best_run.pt"):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    ckpt_path = directory / name

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "rng_state": torch.get_rng_state(),
    }

    torch.save(state, ckpt_path)


def load_checkpoint(*, model, optim, path, name="best_run.pt"):
    ckpt_path = Path(path) / name
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(ckpt["model"])
    # optim.load_state_dict(ckpt["optimizer"])

    return ckpt
