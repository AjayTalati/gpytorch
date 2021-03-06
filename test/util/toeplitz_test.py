import torch
from gpytorch import utils


def test_sym_toeplitz_constructs_tensor_from_vector():
    c = torch.Tensor([1, 6, 4, 5])

    res = utils.toeplitz.sym_toeplitz(c)
    actual = torch.Tensor([
        [1, 6, 4, 5],
        [6, 1, 6, 4],
        [4, 6, 1, 6],
        [5, 4, 6, 1],
    ])

    assert torch.equal(res, actual)


def test_sym_toeplitz_getitem():
    c = torch.Tensor([1, 6, 4, 5])

    actual = torch.Tensor([
        [1, 6, 4, 5],
        [6, 1, 6, 4],
        [4, 6, 1, 6],
        [5, 4, 6, 1],
    ])

    actual_entry = actual[2, 3]
    res_entry = utils.toeplitz.sym_toeplitz_getitem(c, 2, 3)
    assert res_entry == actual_entry


def test_reverse():
    input = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    res = torch.Tensor([
        [3, 2, 1],
        [6, 5, 4],
    ])
    assert torch.equal(utils.reverse(input, dim=1), res)


def test_rcumsum():
    input = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    res = torch.Tensor([
        [6, 5, 3],
        [15, 11, 6],
    ])
    assert torch.equal(utils.rcumsum(input, dim=1), res)
