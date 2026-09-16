"""Source-disjoint RunBugRun split policies for explicit data-scale ablations."""
import hashlib


def select_sources(train_sources, test_sources, *, train_count, development_count,
                   test_count, seed, policy="official-exclusive"):
    """Select source IDs before execution without using correctness outcomes.

    The expanded policy uses more official training sources but still evaluates
    exclusively on problems absent from all official training records. Test
    release records of shared problems are never admitted to held-out data.
    """
    train_sources, test_sources = set(train_sources), set(test_sources)
    if policy not in {"official-exclusive", "expanded-train-exclusive-test"}:
        raise ValueError("unknown RunBugRun split policy")
    if any(type(n) is not int or n < 1 for n in (train_count, development_count, test_count)):
        raise ValueError("split counts must be positive integers")

    def rank(tag, source):
        return hashlib.sha256((str(seed) + "\0" + tag + "\0" + str(source)).encode()).digest()

    available_train = train_sources - test_sources if policy == "official-exclusive" else train_sources
    train_order = sorted(available_train, key=lambda source: rank("train-problem", source))
    test_order = sorted(test_sources - train_sources, key=lambda source: rank("test-problem", source))
    selected = {"train": set(train_order[:train_count]),
                "development": set(train_order[train_count:train_count + development_count]),
                "test": set(test_order[:test_count])}
    expected = {"train": train_count, "development": development_count, "test": test_count}
    if any(len(selected[k]) != n for k, n in expected.items()):
        raise ValueError("requested caps exceed source-disjoint eligible problems")
    if (selected["train"] & selected["development"] or selected["train"] & selected["test"]
            or selected["development"] & selected["test"]):
        raise AssertionError("source split overlap")
    return selected
