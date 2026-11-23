## 1st Run Scores

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

## 2nd Run Scores

What changed: Added pos_weight for BCEWithLogitsLoss
Why changed: The dataset is heavily imbalanced, with around ~11% of readmittion rate. See `First Run`

```
accuracy:  0.6254964257347101
precision:  0.16342608112638285
recall:  0.5937880633373934
f1:  0.25630914826498424
confusion:  [
    [8475, 4991], 
    [ 667,  975]
]
```

## 3rd Run Scores at threshold 0.5 on validation

What changed: Added Trivial and HistGradientBoosted Baseline
Why changed: To establish metrics to beat for my network

```
Trivial Baseline Score
"accuracy": 0.8888816091200944,
"precision": 0.0,
"recall": 0.0,
"f1": 0.0,
"pr_auc": 0.11111839087990566,
"confusion": [
    [13567, 0],
    [1696, 0]
]

HGB Baseline Score

"accuracy": 0.8888816091200944,
"precision": 0.0,
"recall": 0.0,
"f1": 0.0,
"pr_auc": 0.22797038159861208,
"confusion": [
    [13567, 0],
    [1696, 0]
]

My own model scores

"accuracy": 0.6194719255716439,
"precision": 0.16634209672184355,
"recall": 0.6043632075471698,
"f1": 0.2608806312038687,
"pr_auc": 0.20738929722792973,
"confusion": [
    [8430, 5137],
    [671, 1025]
]
```

## 4th Run Scores at threshold 0.9 on validation

What changed: The threshold comparison
Why changed: To experiment the results at different threshold

```
Trivial Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.11111839087990566
Confusion Matrix:  [
    [13567, 0], 
    [1696,  0]
]
HGB Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.22797038159861208
Confusion Matrix:  [
    [13567, 0], 
    [1696, 0]
]
My own model Baseline Scores
Accuracy:  0.8886850553626416
Precision:  0.43478260869565216
Recall:  0.00589622641509434
F1:  0.011634671320535195
PR-AUC:  0.20738929722792973
Confusion Matrix:  [
    [13554, 13], 
    [1686, 10]
]
```