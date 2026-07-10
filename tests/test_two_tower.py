import pytest
import torch
from two_tower import TwoTowerModel

N_USERS = 100
N_ITEMS = 50
EMB_DIM = 16
BATCH = 8


@pytest.fixture
def model():
    m = TwoTowerModel(N_USERS, N_ITEMS, embedding_dim=EMB_DIM)
    m.eval()
    return m


def test_output_shape(model):
    users = torch.randint(0, N_USERS, (BATCH,))
    items = torch.randint(0, N_ITEMS, (BATCH,))
    out = model(users, items)
    assert out.shape == (BATCH,)


def test_single_sample(model):
    user = torch.tensor([0])
    item = torch.tensor([0])
    out = model(user, item)
    assert out.shape == (1,)


def test_no_nan_or_inf(model):
    users = torch.randint(0, N_USERS, (BATCH,))
    items = torch.randint(0, N_ITEMS, (BATCH,))
    out = model(users, items)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_embedding_table_sizes(model):
    assert model.user_embedding.weight.shape == (N_USERS, EMB_DIM)
    assert model.item_embedding.weight.shape == (N_ITEMS, EMB_DIM)
    assert model.user_bias.weight.shape == (N_USERS, 1)
    assert model.item_bias.weight.shape == (N_ITEMS, 1)


def test_global_bias_is_scalar(model):
    assert model.global_bias.shape == ()


def test_deterministic_output(model):
    users = torch.tensor([1, 2, 3])
    items = torch.tensor([4, 5, 6])
    out1 = model(users, items)
    out2 = model(users, items)
    assert torch.allclose(out1, out2)


def test_different_users_different_output(model):
    # With random init, two different users should (almost certainly) produce different scores
    item = torch.tensor([0, 0])
    users = torch.tensor([0, 1])
    out = model(users, item)
    assert out[0].item() != out[1].item()