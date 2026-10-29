"""Tests for Week04 Day01. These are specifications, not answer code."""
import pytest
import torch
from torch.utils.data import DataLoader
from training.data import make_batch, TokenWindowDataset, build_loader


def test_make_batch_shapes_dtype_and_alignment():
    source = list(range(20))
    x, y = make_batch(source, block_size=5, batch_size=7,
                      generator=torch.Generator().manual_seed(123))
    assert x.shape == (7, 5)
    assert y.shape == (7, 5)
    assert x.dtype == y.dtype == torch.long
    assert x.device.type == y.device.type == 'cpu'
    assert torch.equal(x[:, 1:], y[:, :-1])
    for row in range(7):
        start = x[row, 0].item()  # valid because source is range(20)
        assert torch.equal(x[row], torch.arange(start, start+5))
        assert torch.equal(y[row], torch.arange(start+1, start+6))


def test_make_batch_reproducible():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = make_batch(list(range(100)), 6, 12, generator=g1)
    b = make_batch(list(range(100)), 6, 12, generator=g2)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_make_batch_exact_minimum_size():
    x, y = make_batch([9, 8, 7, 6], block_size=3, batch_size=2)
    assert x.tolist() == [[9,8,7],[9,8,7]]
    assert y.tolist() == [[8,7,6],[8,7,6]]


@pytest.mark.parametrize('tokens,block_size,batch_size,error', [
    ([1,2,3], 3, 1, ValueError),
    ([1,2,3], 0, 1, ValueError),
    ([1,2,3], 1, 0, ValueError),
    (torch.tensor([1.0,2.0,3.0]), 1, 1, TypeError),
    (torch.tensor([[1,2],[3,4]]), 1, 1, ValueError),
    ([-1,1,2,3], 2, 1, ValueError),
])
def test_make_batch_bad_inputs(tokens, block_size, batch_size, error):
    with pytest.raises(error):
        make_batch(tokens, block_size, batch_size)


def test_dataset_length_and_all_windows():
    ds = TokenWindowDataset([0,1,2,3,4,5,6], block_size=3)
    assert len(ds) == 4
    assert [ds[i][0].tolist() for i in range(4)] == [
        [0,1,2], [1,2,3], [2,3,4], [3,4,5]
    ]
    assert [ds[i][1].tolist() for i in range(4)] == [
        [1,2,3], [2,3,4], [3,4,5], [4,5,6]
    ]
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.dtype == y.dtype == torch.long
        assert torch.equal(x[1:], y[:-1])


def test_dataset_index_and_initialization_errors():
    ds = TokenWindowDataset([0,1,2,3], 2)
    with pytest.raises(IndexError): _ = ds[-1]
    with pytest.raises(IndexError): _ = ds[len(ds)]
    with pytest.raises(TypeError): _ = ds[0.5]
    with pytest.raises(ValueError): TokenWindowDataset([1,2], 2)
    with pytest.raises(ValueError): TokenWindowDataset([1,2,3], 0)


def test_loader_sequential_batches():
    loader = build_loader(list(range(9)), block_size=3,
                          batch_size=2, shuffle=False, drop_last=False)
    assert isinstance(loader, DataLoader)
    assert len(loader.dataset) == 6
    batches = list(loader)
    assert len(batches) == 3
    x, y = batches[0]
    assert x.shape == y.shape == (2,3)
    assert x.tolist() == [[0,1,2],[1,2,3]]
    assert y.tolist() == [[1,2,3],[2,3,4]]


def test_loader_drop_last():
    keep = build_loader(list(range(8)), 3, 2,
                        shuffle=False, drop_last=False)
    drop = build_loader(list(range(8)), 3, 2,
                        shuffle=False, drop_last=True)
    assert [x.shape[0] for x, _ in keep] == [2,2,1]
    assert [x.shape[0] for x, _ in drop] == [2,2]


def test_loader_deterministic_shuffle():
    args = dict(tokens=list(range(40)), block_size=3,
                batch_size=4, shuffle=True)
    a = build_loader(**args, generator=torch.Generator().manual_seed(77))
    b = build_loader(**args, generator=torch.Generator().manual_seed(77))
    assert [x.tolist() for x,_ in a] == [x.tolist() for x,_ in b]


def test_loader_invalid_arguments():
    with pytest.raises(ValueError):
        build_loader(list(range(10)), 3, batch_size=0)
    with pytest.raises(ValueError):
        build_loader(list(range(10)), 3, batch_size=2, num_workers=-1)
