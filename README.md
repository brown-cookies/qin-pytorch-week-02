# A classifier that has to make a decision

## Table of Contents

## Problem

## Datasets

## Reproducibility

## Data after training

## Error Analysis

### First run didn't predict anything set at threshold 0.5 (Scored on test)

**Scores**
```
accuracy:  0.8913158591474716
precision:  0.0
recall:  0.0
f1:  0.0
confusion:  [
    [13466     0]
    [ 1642     0]
]
```

**Observed:** accuracy equals the share of non-readmitted patients TP and FP are both 0, so no patient scored ≥ 0.5.

**Hypothesis:** with ~11% positives and an unweighted loss, predicted
probabilities stay near the base rate, below 0.5.

**Solution:** add `pos_weight` to `BCEWithLogitsLoss`, so missed positives
cost more and scores move upward. See `LOGS.md Run 2` for the results.


## Diagnosis

### 1. Run 1, 2, 3 at 0.5, 0.9, 0.1 Threshold (Testing the extreme cases)

**Main finding:** The extreme thresholds do not show which model is better.

**1. Trivial baseline**
- It outputs one constant score (about 0.111).
- At 0.5 and 0.9 it predicts all negative, so recall = 0 and accuracy = 0.8889 (the negative-class rate).
- At 0.1 it predicts all positive, so recall = 1.0 and F1 = 0.200.
- This gives the floor: **F1 = 0.200, PR-AUC = 0.111**.

**2. HGB baseline**
- At 0.5 and 0.9 it predicts zero positives.
- At 0.1 it works: precision 0.151, recall 0.736, F1 0.251.
- Cause: HGB was trained without class weights, so its probabilities stay low, near the 11% base rate. Thresholds of 0.5 and 0.9 are never reached.

**3. My model**

| Threshold | Precision | Recall | F1 | Behavior |
|---|---|---|---|---|
| 0.1 | 0.111 | 1.000 | 0.200 | predicts everything positive (same as trivial) |
| 0.5 | 0.166 | 0.604 | 0.261 | useful split |
| 0.9 | 0.435 | 0.006 | 0.012 | predicts almost nothing positive |

- At 0.1, every score is above the threshold, so my model is similar to the trivial baseline.
- At 0.9, almost no scores pass, so recall collapses (only 10 of 1696 positives found).
- Only 0.5 gives a real split. My probabilities are bunched in the middle.

**5. Conclusion**
- My model (F1 0.261 at 0.5) is slightly above HGB's best of these runs (F1 0.251 at 0.1), but its PR-AUC (0.207) is below HGB (0.228).

## Ways of Handling the Imbalanced (What I did)

### 1. Weighted Loss

### 2. 

## AI Usage
| File/Function | How it was used |
|---|---|
| `src/data` | Asked AI for guidance on how to load, drop, fill, transform, encode the dataset. The code is AI generated, every action it does is what I specify, As I'm not profeccient at using `Pandas` |
| `tests/test_data` |  |
| `tests/test_checkpoint` |  |