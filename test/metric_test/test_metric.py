import pytest
import torch

from lib.metric import metric


@pytest.fixture
def data():
    pred = torch.Tensor([
        [0.1, 0.7, 0.2],
        [0.7, 0.1, 0.2],
        [0.1, 0.2, 0.7]
    ])
    y = torch.Tensor([0, 0, 0])
    return pred, y


def test_accuracy(data):
    pred, y = data
    actual = metric.accuracy(pred, y)
    assert actual == 1. / 3


def test_correct(data):
    pred, y = data
    actual = metric.correct(pred, y)
    expect = torch.Tensor([False, True, False])
    assert torch.all(actual.eq(expect))


def test_incorrect(data):
    pred, y = data
    actual = metric.incorrect(pred, y)
    expect = torch.Tensor([True, False, True])
    assert torch.all(actual.eq(expect))
