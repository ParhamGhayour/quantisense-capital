"""
Quantisense Capital
Phase 2 AI Training
"""

from sklearn.model_selection import TimeSeriesSplit


def temporal_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)

    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:]
    )


def walk_forward_splits(X, y, n_splits=5):
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in splitter.split(X):
        yield (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx]
        )
def purged_walk_forward_splits(X, y, horizon=5, n_splits=5):
    """
    Walk-forward validation with a purge gap.

    The final `horizon` observations of each training window
    are removed so their future-return targets cannot overlap
    the following test window.
    """

    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in splitter.split(X):
        if len(train_idx) <= horizon:
            continue

        train_idx = train_idx[:-horizon]

        yield (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx]
        )