# Illicit Bitcoin transaction detection (BT4012 Kaggle competition, 2026)

Predict the probability that each transaction in periods 36–49 is illicit, from labelled transactions in periods 1–35 and the transaction graph. The metric is ROC AUC.

Both final submissions are built end to end by [`e1123147.ipynb`](e1123147.ipynb):

| File | Public ROC AUC |
|---|---|
| `outputs/final1_main_plus_25pct_labelled_catboost.csv` (top submission) | **0.96823** |
| `outputs/final2_hedge_main_and_labelled_only_50_50.csv` | 0.96734 |

## Repository

```
├── e1123147.ipynb     # final models: preprocessing, tuning, feature engineering, validation, submission CSVs
├── utils.py           # small helpers used by the notebook (loading, rank blending, saving)
├── requirements.txt   # dependencies (versions used for the scored files)
├── tuning/            # Optuna trial logs and best trials behind the CatBoost / LightGBM parameters
├── AI_DECLARATION.txt # how AI assistance was used
└── data/              # competition CSVs go here (not committed, see data/README.md)
```

## Run it

```bash
pip install -r requirements.txt
# put train.csv, test.csv, txs_edgelist.csv and sample_submission.csv in data/
jupyter nbconvert --to notebook --execute e1123147.ipynb   # or run all cells in Jupyter
```

A full run takes about 45 minutes on 4 CPU threads (about 30 without the validation section). Set `RUN_TUNING = True` to rerun the Optuna searches as well (about 20 more minutes). It writes the two files above to `outputs/`.

With the pinned versions in `requirements.txt`, the two files are identical to the ones uploaded to Kaggle.

## Model

1. **Unknown labels are used.** About 78% of training transactions are unlabelled. They are trained on as licit, at 0.3× the weight of a labelled row.
2. **Recent periods count more.** Each row is weighted by `exp((t − 35) / 10)`.
3. **Two stages per model.** Stage 1 scores every transaction from its 165 features; training rows are scored out of fold. Stage 2 adds the max / mean / count of each transaction's *labelled* graph neighbours' stage-1 scores, split by direction, and retrains.
4. **Blend.** CatBoost and LightGBM are rank-averaged into the main model (public 0.96816).
5. **Finals.** Final 1 adds 25% of a CatBoost trained the same way on labelled rows only. Final 2 is a 50/50 hedge between the main model and the labelled-only CatBoost + LightGBM. The hedge is safer if the unknowns mislead on new fraud types (see the validation section of the notebook), and costs only 0.0009 on public.

How the public score got there:

| Step | Public ROC AUC |
|---|---|
| Tuned CatBoost on labelled rows | 0.94944 |
| + graph neighbour-score stage | 0.95252 |
| + recency weights (tau 10) | 0.96009 |
| + LightGBM, rank-blended | 0.96168 |
| + unknown rows as licit, weight 0.3 | 0.96816 |
| + 25% labelled-only CatBoost | **0.96823** |

The notebook explains each step. It also reports a validation finding: on older time windows with new fraud types, the unknowns-as-licit idea scores *worse*. That is discussed in section 5 of the notebook.

## Key decisions

| Decision | What I chose | Alternatives tried (public ROC AUC unless noted) | Why |
|---|---|---|---|
| How to validate | Forward-in-time splits (train on earlier periods, score later ones); public leaderboard as the final judge | Random splits give about 0.999 for every model | The test periods come after all training periods. Random splits leak the future, and they can't tell models apart. |
| Main model family | Gradient-boosted trees: CatBoost, later + LightGBM | LightGBM alone 0.94606, random forest 0.932, logistic regression, neural nets and a graph neural network (both clearly weaker locally), XGBoost as a third model 0.96760 | Tuned CatBoost was the strongest single model (0.94944). LightGBM was about as good but made different mistakes, so blending the two helped (0.96168). |
| Use of the graph | A second stage with labelled neighbours' model scores | Raw neighbour feature averages, degrees and paths, a GNN, 2-hop scores through unlabelled nodes (0.96134) | Neighbour *scores* moved the leaderboard (0.94983 → 0.95252); the rest didn't. Only labelled neighbours are used so that training and test see the graph the same way. |
| Time drift | Recency weights `exp((t − 35) / 10)` | tau 7 / 20 / 30, a separate tau before and after period 24 (0.95995–0.96110), learned "test-likeness" weights (much worse locally) | Recent periods resemble the test most, but old periods still hold useful fraud examples. Tau 10 balanced the two. |
| Unlabelled rows | Use them as licit at weight 0.3 | Dropping them (0.96168), full weight (0.967), weights 0.15 / 0.6 | The largest single gain (+0.0065). The reduced weight reflects that some unknowns are undiscovered fraud. |
| Hyperparameters | Optuna on forward folds, labelled rows (CatBoost 30 trials, LightGBM 25) | Shallower / more regularised CatBoost 0.94780, a shallow bag 0.94360 | Lower-capacity models consistently lost on the leaderboard. |
| PCA | Left out | PCA-only model about 0.06 AUC lower locally; 30 components added to stage 2: 0.95967 | Tree models already handle the 165 correlated features. PCA keeps the directions with the most variance, not the most fraud signal, and those shift over time. |
| Other features left out | Weight of evidence / information value, IsolationForest anomaly scores, dropping local or aggregated feature groups | WoE hurt trees locally (0.950 vs 0.961), anomaly features 0.95997 / 0.96080, dropping a feature group lost 0.03–0.04 locally | None beat the plain 165 features plus graph scores. |
| Final picks | Final 1: main model + 25% labelled-only CatBoost (0.96823). Final 2: a 50/50 hedge with the labelled-only blend (0.96734) | Two near-identical unknowns-based files (0.96823 + 0.96816) | Local windows with new fraud disagree strongly with the leaderboard about the unknowns. The hedge costs only 0.0009 on public but protects the private score if that disagreement is real. |

## AI usage

I used Claude (Anthropic) as a coding assistant for writing code, running experiments and drafting documentation. Modelling decisions, submissions and the final selection are my own. See [`AI_DECLARATION.txt`](AI_DECLARATION.txt).
