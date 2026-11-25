import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.model import ReadmittanceModel


N_FEATURES = 8


def test_round_trip_weights_and_bias_match(tmp_path):
    torch.manual_seed(0)

    model = ReadmittanceModel(N_FEATURES)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

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
    new_optim = torch.optim.SGD(model.parameters(), lr=0.01)

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
