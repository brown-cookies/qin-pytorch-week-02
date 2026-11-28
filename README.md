# A classifier that has to make a decision

## Table of Contents

- [Problem](#problem)
- [Datasets](#datasets)
- [Reproducibility](#reproducibility)
- [Data after training](#data-after-training)
- [Error Analysis](#error-analysis)
  - [First run didn't predict anything set at threshold 0.5 (Scored on test)](#first-run-didnt-predict-anything-set-at-threshold-05-scored-on-test)
- [Diagnosis](#diagnosis)
  - [1. Run 1, 2, 3 at 0.5, 0.9, 0.1 Threshold (Testing the extreme cases)](#1-run-1-2-3-at-05-09-01-threshold-testing-the-extreme-cases)
  - [2. Run 8, 9, 10, 11 at threshold 0.5 (Scored on validation)](#2-run-8-9-10-11-at-threshold-05-scored-on-validation)
- [Leakage Audit](#leakage-audit)
  - [1. Patient ID Grouping `patient_nbr`](#1-patient-id-grouping-patient_nbr)
- [Ways of Handling the Imbalanced (What I did)](#ways-of-handling-the-imbalanced-what-i-did)
  - [1. Weighted Loss](#1-weighted-loss)
  - [2. WeightedRandomSampler](#2-weightedrandomsampler)
- [AI Usage](#ai-usage)

## Problem

A team has more cases than it can deal with. Most of them do not matter and a few matter a great deal. The team can only act on some of them, or acting wrongly costs one amount and missing a real case costs far more. They have historical data, and they have asked you one question: which cases should we act on, and what do we get for it?

You are to build a classifier on a real, imbalanced table of data and turn its scores into that decision. "The model is 97% accurate" is not an answer here: on a table like this, a model that always says no can score that too. Your answer is a statement the team can act on, with a number behind it that survives a reviewer asking how you got it.

This week is also about the training loop itself. Runs crash, sessions time out, and results that look good are often wrong. So your loop has to be disciplined enough that you can stop it and carry on, compare one run with another, and say why a result is bad when it is.



## Datasets

[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us-hospitals-for-years-1999-2008)
from the UCI Machine Learning Repository (CC BY 4.0). The project downloads an
unchanged copy from the GitHub Release (tag `0.1.0`) into `datasets/`. See `ATTRIBUTION.md` for reference.

| | |
|---|---|
| Rows | 101,766 (one row per hospital visit) |
| Unique patients | ~71k (`patient_nbr`), so some patients have more than one visit |
| Target | `readmitted` = `<30` becomes 1, everything else becomes 0 |
| Positive rate | about 11% readmitted within 30 days |
| Missing values | Marked with `?` in the CSV. |
| Features after preparation | 276 columns (after encoding and one-hot encoding) |

**Class balance**

| Class | Share |
|---|---|
| Not readmitted within 30 days (0) |  ~89% |
| Readmitted within 30 days (1) | ~11% |

## Reproducibility

### Quick Setup
Needs Python 3.12. Everything runs on CPU, no GPU needed.

```
git clone https://github.com/brown-cookies/qin-pytorch-week-02.git
cd qin-pytorch-week-02
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

The dataset downloads itself into `datasets/` the first time it is needed.
It is an unchanged copy from the GitHub Release (tag `0.1.0`), checked
against a SHA-256 checksum. See `ATTRIBUTION.md` for the source and licence.

### Re-running
1. Split. Grouped by `patient_nbr`, `seed = 42` (in `src/config.py`), so two runs give the same indices and no patient appears in two sets.

```
python -m src.main split --out runs/split.json
```

2. Smoke training run. 10 epochs, batch size 46. It only checks that the training code runs, about `<N>` minutes on CPU.

```
python -m src.main train --smoke --out runs/smoke
```

3. Full training. Writes `runs/manual/best_run.pt` (best validation loss) and `last_run.pt`.

```
python -m src.main train --out runs/train
```

4. Evaluate on the test set. Loads the checkpoint only.

```
python -m src.main evaluate --checkpoint runs/manual/best_run.pt --metrics-out runs/metrics.json
```

Expected: `<metric>` = `<value>` (±`<tolerance>`, because `<reason>`).

To train the three seeds (42, 76, 127) run `python -m src.main run`.

## Data after training across three seeds (42, 76, 127)

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

**Conclusion:** My model (F1 0.261 at 0.5) is slightly above HGB's best of these runs (F1 0.251 at 0.1), but its PR-AUC (0.207) is below HGB (0.228).

### 2. Run 8, 9, 10, 11 at threshold 0.5 (Scored on validation)

**Main Finding:** These runs test two ways of handling the imbalance, with everything else the same.

| Run | What changed |
|---|---|
| Run 8 | No weighted loss |
| Run 9 | Weighted loss (`pos_weight`) |
| Run 10 | No WeightedRandomSample (baseline for experiment 2) |
| Run 11 | WeightedRandomSampler on the train loader, plain loss |

| My model at 0.5 (validation) | Run 8 | Run 9 | Run 10 | Run 11 |
|---|---|---|---|---|
| Accuracy | 0.8888816091200944 | 0.6194719255716439 | 0.8888816091200944 | 0.6422721614361527 |
| Precision | 0.0 | 0.16634209672184355 | 0.0 | 0.16678470254957506 |
| Recall | 0.0 | 0.6043632075471698 | 0.0 | 0.5554245283018868 |
| F1 | 0.0 | 0.2608806312038687 | 0.0 | 0.2565359477124183 |
| PR-AUC | 0.1989840003072726 | 0.20738929722792973 | 0.1989840003072726 | 0.20123723907013816 |
| Confusion matrix | [[13567, 0], [1696, 0]] | [[8430, 5137], [671, 1025]] | [[13567, 0], [1696, 0]] | [[8861, 4706], [754, 942]] |

**Conclusion:** Weighted loss performs slightly better than the WeightedRandomSampler.
It has higher recall (0.604 vs 0.555), higher F1 (0.261 vs 0.257), higher PR-AUC
(0.207 vs 0.201), and misses fewer readmitted patients (671 vs 754).

### 3. Run 13, 14, 15 at threshold 0.5 (Scored on validation)

**Main Finding:** These runs checks whether changing batch size have impact on PR-AUC. Lowering the batch size improves the PR-AUC

**DEFAULT Batch Size:** 512 (Run 9)

| Run | What changed |
|---|---|
| Run 13 | Increased to 1024 |
| Run 14 | Decreased to 256 |
| Run 15 | Decreased to 128 |

| My model at 0.5 (validation) | Run 9 (default) | Run 13 | Run 14 | Run 15 |
|---|---|---|---|---|
| Accuracy | 0.6194719255716439 | 0.6441721810915285 | 0.6244512874271113 | 0.5897267902771408 |
| Precision | 0.16634209672184355 | 0.16514254975793438 | 0.17122841316389703 | 0.16544548651817115 |
| Recall | 0.6043632075471698 | 0.5430424528301887 | 0.6196933962264151 | 0.6656839622641509 |
| F1 | 0.2608806312038687 | 0.2532655025436546 | 0.26831758999234107 | 0.2650234741784038 |
| PR-AUC | 0.20738929722792973 | 0.200104714437276 | 0.21910716656522888 | 0.22815517456710088 |
| Confusion matrix | [[8430, 5137], [671, 1025]] | [[8911, 4656], [775, 921]] | [[8480, 5087], [645, 1051]] | [[7872, 5695], [567, 1129]] |

**Conclusion:** PR-AUC rises as the batch size goes down, Batch 128 matches the Gradient-Boosted baseline (0.2280), with a gap of about 0.0002. This is
far better than having a gap of about 0.02.

## Leakage Audit

### 1. Patient ID Grouping `patient_nbr`
```
rows: 101766
unique patients: 71518
```

**Observation** The dataset has ~101k rows but the `patient_nbr` has only around ~71k count. So some patients
have a multiple visit. After researching, if not grouped properly, the model can just memorize those and inflates the
test scores.

**What I did** Split by patients, not by rows. All rows of the same `patient_nbr` goes to same set (train, val, test).

## Ways of Handling the Imbalanced (What I did)

### 1. Weighted Loss

### Run scores without the weighted loss (scored on validation) at threshold 0.5

```
My own model Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.1989840003072726
Confusion Matrix:  [[13567, 0], [1696, 0]]
```
### Run scores with weighted loss (scored on validation) at threshold 0.5

```
My own model Baseline Scores
Accuracy:  0.6194719255716439
Precision:  0.16634209672184355
Recall:  0.6043632075471698
F1:  0.2608806312038687
PR-AUC:  0.20738929722792973
Confusion Matrix:  [[8430, 5137], [671, 1025]]
```

**Observation**

Without the weighted loss, my model predicted nobody as readmitted. It had
0 recall and 0 precision. The accuracy of 0.889 looks good, but it only
means the model said "not readmitted" for everyone, and about 89% of patients
are not readmitted. With the weighted loss, the model started finding readmitted patients

### 2. WeightedRandomSampler

### Run scores without WeightedRandomSampler (scored on validation) at threshold 0.5

```
My own model Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.1989840003072726
Confusion Matrix:  [[13567, 0], [1696, 0]]
```

### Run scores with WeightedRandomSampler (scored on validation) at threshold 0.5

```
My own model Baseline Scores
Accuracy:  0.6422721614361527
Precision:  0.16678470254957506
Recall:  0.5554245283018868
F1:  0.2565359477124183
PR-AUC:  0.20123723907013816
Confusion Matrix:  [[8861, 4706], [754, 942]]
```
**Observation**

The WeightedRandomSampler fixed the same problem as the weighted loss. The model
no longer predicts everyone as "not readmitted".

| My model at 0.5 (validation) | Without weight | Weighted loss | WeightedRandomSampler |
|---|---|---|---|
| Accuracy | 0.8888816091200944 | 0.6194719255716439 | 0.6422721614361527 |
| Precision | 0.0 | 0.16634209672184355 | 0.16678470254957506 |
| Recall | 0.0 | 0.6043632075471698 | 0.5554245283018868 |
| F1 | 0.0 | 0.2608806312038687 | 0.2565359477124183 |
| PR-AUC | 0.1989840003072726 | 0.20738929722792973 | 0.20123723907013816 |
| Confusion matrix | [[13567, 0], [1696, 0]] | [[8430, 5137], [671, 1025]] | [[8861, 4706], [754, 942]] |

## AI Usage
| File/Function | How it was used |
|---|---|
| `src/data` | Asked AI for guidance on how to load, drop, fill, transform, encode the dataset. The code is AI generated, every action it does is what I specify, As I'm not profeccient at using `Pandas` |
| `tests/test_data` | Asked AI to help me what kind of test should I write, with references of `torch/week-01-solution`  |
| `tests/test_checkpoint` | Asked AI for guidance why the test keeps failing and help me diagnose the problem |
| `src/helper` | Asked AI for formatting the `print_numbers` into much more readable data |