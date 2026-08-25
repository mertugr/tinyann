"""Python bindings: exact cosine Index. Includes crash / bad-input cases."""

import math

import pytest

import tinyann


def test_add_search_remove_cosine() -> None:
    index = tinyann.Index(4, "cosine")
    assert index.dim == 4
    assert index.metric == "cosine"
    assert len(index) == 0
    index.add(1, [1.0, 0.0, 0.0, 0.0])
    index.add(2, [0.0, 1.0, 0.0, 0.0])
    hits = index.search([0.9, 0.1, 0.0, 0.0], k=2)
    assert [item[0] for item in hits] == [1, 2]
    assert hits[0][1] > hits[1][1]
    assert index.remove(1) is True
    assert index.contains(1) is False
    assert [item[0] for item in index.search([0.9, 0.1, 0.0, 0.0], k=2)] == [2]


def test_filtered_search() -> None:
    index = tinyann.Index(2)
    index.add(1, [1.0, 0.0])
    index.add(2, [0.99, 0.01])
    hits = index.search([1.0, 0.0], k=1, allow_ids=[1])
    assert [item[0] for item in hits] == [1]


def test_duplicate_id_raises() -> None:
    index = tinyann.Index(2)
    index.add(1, [1.0, 0.0])
    with pytest.raises(ValueError, match="duplicate"):
        index.add(1, [0.0, 1.0])


def test_wrong_dim_does_not_crash() -> None:
    index = tinyann.Index(2)
    with pytest.raises(ValueError, match="dimension"):
        index.add(1, [1.0])
    with pytest.raises(ValueError, match="dimension"):
        index.search([1.0], k=1)
    assert len(index) == 0


def test_nan_and_inf_rejected() -> None:
    index = tinyann.Index(2)
    with pytest.raises(ValueError):
        index.add(1, [math.nan, 0.0])
    with pytest.raises(ValueError):
        index.add(1, [math.inf, 0.0])
    index.add(1, [1.0, 0.0])
    with pytest.raises(ValueError):
        index.search([math.nan, 0.0], k=1)
    assert len(index) == 1


def test_empty_and_k_zero() -> None:
    index = tinyann.Index(2)
    assert index.search([1.0, 0.0], k=3) == []
    assert index.search([1.0, 0.0], k=0) == []
    index.add(1, [1.0, 0.0])
    assert index.search([1.0, 0.0], k=0) == []


def test_remove_missing_and_zero_dim() -> None:
    index = tinyann.Index(2)
    assert index.remove(99) is False
    with pytest.raises(ValueError):
        tinyann.Index(0)


def test_only_cosine_is_exposed() -> None:
    with pytest.raises(ValueError, match="cosine"):
        tinyann.Index(2, "euclidean")
