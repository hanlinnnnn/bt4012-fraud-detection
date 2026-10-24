"""Small helpers used by e1123147.ipynb: file loading, blending utilities and saving submissions.
All modelling decisions are in the notebook itself."""
import io

import pandas as pd

FEATURES = [f'feat_{i}' for i in range(1, 166)]


def load_data(data_dir):
    # features as float32: every scored submission was built this way
    dtypes = {f: 'float32' for f in FEATURES}
    train = pd.read_csv(data_dir / 'train.csv', dtype=dtypes)
    test = pd.read_csv(data_dir / 'test.csv', dtype=dtypes)
    edges = pd.read_csv(data_dir / 'txs_edgelist.csv')
    return train, test, edges


def rank_pct(x):
    # scores -> percentile ranks in (0, 1], so models on different scales can be averaged
    return pd.Series(x).rank(pct=True).values


def as_written(p):
    # scores rounded exactly like a csv saved with 12 significant digits
    # (the original blends were computed from saved files, so this keeps results identical)
    csv = pd.DataFrame({'target': p}).to_csv(index=False, float_format='%.12g')
    return pd.read_csv(io.StringIO(csv)).target.values


def save_submission(path, index, target):
    pd.DataFrame({'index': index, 'target': target}).to_csv(path, index=False, float_format='%.12g')

