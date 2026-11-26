from dataclasses import replace

import pandas as pd
import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import Config
from src.loader import make_loader
from src.model import ReadmittanceModel
from src.train import train_model


N_FEATURES = 8


def test_round_trip_weights_and_bias_match(tmp_path):
    torch.manual_seed(0)

    model = ReadmittanceModel(N_FEATURES)
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    save_checkpoint(
        epoch=1,
        model=model,
        optim=optim,
        history={},
        best_val_loss=0.5,
        path=tmp_path
    )

    torch.manual_seed(1)
    new_model = ReadmittanceModel(N_FEATURES)
    new_optim = torch.optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

    load_checkpoint(
        model=new_model,
        optim=new_optim,
        path=tmp_path
    )

    weights = model.state_dict()
    new_weights = new_model.state_dict()

    assert model.state_dict().keys() == new_model.state_dict().keys()
    assert torch.equal(weights["net.0.weight"], new_weights["net.0.weight"])
    assert torch.equal(weights["net.0.bias"], new_weights["net.0.bias"])
    assert torch.equal(weights["net.2.weight"], new_weights["net.2.weight"])
    assert torch.equal(weights["net.2.bias"], new_weights["net.2.bias"])
    assert torch.equal(weights["net.4.weight"], new_weights["net.4.weight"])
    assert torch.equal(weights["net.4.bias"], new_weights["net.4.bias"])


def test_resume_run_match_with_uninterrupted_run(tmp_path, cfg):
    # i = interrupted
    # u = uniterrupted
    # r = resume interrupted

    torch.manual_seed(0)

    lr = 0.01

    def init_everything(seed, lr):
        torch.manual_seed(seed)

        X = torch.randn(200, N_FEATURES)
        true_w = torch.randn(N_FEATURES)
        y = (torch.rand(200) <
             torch.sigmoid(X @ true_w - 2.0)).float()

        X_df = pd.DataFrame(X.numpy())
        y_np = y.numpy()

        train_loader = make_loader(
            X_df.iloc[:140], y_np[:140], batch_size=32, shuffle=True
        )

        val_loader = make_loader(X_df.iloc[140:], y_np[140:], batch_size=32)

        model = ReadmittanceModel(N_FEATURES)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.BCEWithLogitsLoss()

        return model, optimizer, criterion, train_loader, val_loader

    i_model, i_optim, i_criterion, i_train_loader, i_val_loader = init_everything(
        0, lr
    )

    print(i_optim.state_dict())

    train_model(
        model=i_model,
        train_loader=i_train_loader,
        val_loader=i_val_loader,
        optimizer=i_optim,
        criterion=i_criterion,
        resume=False,
        checkpoint_path=tmp_path / "interrupted",
        cfg=replace(cfg, epochs=5)
    )

    print(i_optim.state_dict())

    u_model, u_optim, u_criterion, u_train_loader, u_val_loader = init_everything(
        0, lr
    )

    train_model(
        model=u_model,
        train_loader=u_train_loader,
        val_loader=u_val_loader,
        optimizer=u_optim,
        criterion=u_criterion,
        resume=False,
        checkpoint_path=tmp_path / "unterrupted",
        cfg=replace(cfg, epochs=10)
    )

    r_model, r_optim, r_criterion, r_train_loader, r_val_loader = init_everything(
        0, lr
    )

    train_model(
        model=r_model,
        train_loader=r_train_loader,
        val_loader=r_val_loader,
        optimizer=r_optim,
        criterion=r_criterion,
        resume=True,
        checkpoint_path=tmp_path / "interrupted",
        cfg=replace(cfg, epochs=10)
    )

    # gap = max((u - r).abs().max().item()
    #           for u, r in zip(u_model.state_dict().values(), r_model.state_dict().values()))

    # print("max abs diff:", gap)

    for u, r in zip(u_model.state_dict().values(), r_model.state_dict().values()):
        assert torch.allclose(u, r)
