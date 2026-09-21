import hashlib

import pytest

from pbpf.apbpf.rbr_splits import select_sources


def test_default_split_matches_original_ranking():
    train = {f"p{i}" for i in range(30)}
    test = {f"p{i}" for i in range(20, 40)}
    result = select_sources(train, test, train_count=10, development_count=5, test_count=8, seed=1701)
    def rank(tag, key):
        return hashlib.sha256(f"1701\0{tag}\0{key}".encode()).digest()
    ordered = sorted(train - test, key=lambda key: rank("train-problem", key))
    assert result["train"] == set(ordered[:10])
    assert result["development"] == set(ordered[10:15])


def test_expanded_training_keeps_all_held_out_sources_absent_from_official_train():
    train, test = set(range(30)), set(range(20, 40))
    result = select_sources(train, test, train_count=20, development_count=10,
                            test_count=10, seed=1701, policy="expanded-train-exclusive-test")
    assert result["test"] == set(range(30, 40))
    assert result["train"] | result["development"] == train
    assert not result["train"] & result["development"]
    assert not (result["train"] | result["development"]) & result["test"]


def test_impossible_exclusive_population_is_rejected():
    with pytest.raises(ValueError, match="caps exceed"):
        select_sources(set(range(30)), set(range(20, 40)), train_count=20,
                       development_count=10, test_count=10, seed=1701)
