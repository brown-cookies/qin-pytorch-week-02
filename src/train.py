import torch


from src.checkpoint import save_checkpoint, load_checkpoint
from src.config import Config


def _calculate_val_loss_acc(*, model, val_loader, criterion, cfg: Config):
    """For calculating validation loss, and accuracy per epoch"""
    model.eval()

    total_val_loss = 0.0
    total_val_acc = 0
    val_n = 0

    with torch.no_grad():

        for X, y in val_loader:
            z = model(X)
            loss = criterion(z, y)
            probs = torch.sigmoid(z)
            preds = (probs >= 0.5).float()
            acc = (preds == y).sum().item()
            total_val_loss += (loss.item() * X.size(0))
            total_val_acc += acc
            val_n += X.size(0)

    val_loss = total_val_loss / val_n
    val_acc = total_val_acc / val_n

    return val_loss, val_acc


def train_model(*, model, train_loader, val_loader, optimizer, criterion, resume=False, checkpoint_path, cfg: Config):
    """Train the model"""
    epochs = cfg.epochs
    lr = cfg.learning_rate
    seed = cfg.seed

    best_val_loss = float("inf")

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []

    start = 0

    if resume:
        ckpt = load_checkpoint(
            model=model,
            optim=optimizer,
            path=checkpoint_path,
            name="last_run.pt"
        )

        last_epoch = ckpt["epoch"]
        train_loss_per_epoch = ckpt["history"]["train_loss_per_epoch"]
        val_loss_per_epoch = ckpt["history"]["val_loss_per_epoch"]
        val_acc_per_epoch = ckpt["history"]["val_acc_per_epoch"]
        best_val_loss = ckpt["best_val_loss"]

        start = last_epoch + 1

    for epoch in range(start, epochs):
        model.train()

        total_train_loss = 0.0
        train_n = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            z = model(X)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

            total_train_loss += (loss.item() * X.size(0))
            train_n += X.size(0)

        train_loss = total_train_loss / train_n
        val_loss, val_acc = _calculate_val_loss_acc(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            cfg=cfg
        )

        train_loss_per_epoch.append(train_loss)
        val_loss_per_epoch.append(val_loss)
        val_acc_per_epoch.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                epoch=epoch,
                model=model,
                optim=optimizer,
                history={
                    "train_loss_per_epoch": train_loss_per_epoch,
                    "val_loss_per_epoch": val_loss_per_epoch,
                    "val_acc_per_epoch": val_acc_per_epoch,
                },
                best_val_loss=best_val_loss,
                path=checkpoint_path

            )

        save_checkpoint(
            epoch=epoch,
            model=model,
            optim=optimizer,
            history={
                "train_loss_per_epoch": train_loss_per_epoch,
                "val_loss_per_epoch": val_loss_per_epoch,
                "val_acc_per_epoch": val_acc_per_epoch,
            },
            best_val_loss=best_val_loss,
            path=checkpoint_path,
            name="last_run.pt"
        )

    return {
        "train_loss_per_epoch": train_loss_per_epoch,
        "val_loss_per_epoch": val_loss_per_epoch,
        "val_acc_per_epoch": val_acc_per_epoch,
    }
