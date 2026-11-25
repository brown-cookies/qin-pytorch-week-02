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

## 5th Run Scores at threshold 0.1 on validation

What changed: The threshold comparison
Why changed: To experiment the results at different threshold

```
Trivial Baseline Scores
Accuracy:  0.11111839087990566
Precision:  0.11111839087990566
Recall:  1.0
F1:  0.2000117931481809
PR-AUC:  0.11111839087990566
Confusion Matrix:  [[0, 13567], [0, 1696]]
HGB Baseline Scores
Accuracy:  0.511957020245037
Precision:  0.15137559083747423
Recall:  0.7364386792452831
F1:  0.25113099426962904
PR-AUC:  0.22797038159861208
Confusion Matrix:  [[6565, 7002], [447, 1249]]
My own model Baseline Scores
Accuracy:  0.11111839087990566
Precision:  0.11111839087990566
Recall:  1.0
F1:  0.2000117931481809
PR-AUC:  0.20738929722792973
Confusion Matrix:  [[0, 13567], [0, 1696]]
```

## 6th Run Scores at threshold 0.25 on validation

What changed: The threshold comparison
Why changed: To experiment the results at different threshold

```
Trivial Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.11111839087990566
Confusion Matrix:  [[13567, 0], [1696, 0]]
HGB Baseline Scores
Accuracy:  0.8871781432221713
Precision:  0.44921875
Recall:  0.06780660377358491
F1:  0.11782786885245902
PR-AUC:  0.22797038159861208
Confusion Matrix:  [[13426, 141], [1581, 115]]
My own model Baseline Scores
Accuracy:  0.12337024176112167
Precision:  0.11234240212342401
Recall:  0.9982311320754716
F1:  0.2019563402123345
PR-AUC:  0.20738929722792973
Confusion Matrix:  [[190, 13377], [3, 1693]]
```

## 7th Run Scores at threshold 0.75 on validation

What changed: The threshold comparison
Why changed: To experiment the results at different threshold

```
Trivial Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.11111839087990566
Confusion Matrix:  [[13567, 0], [1696, 0]]
HGB Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.22797038159861208
Confusion Matrix:  [[13567, 0], [1696, 0]]
My own model Baseline Scores
Accuracy:  0.8835746576688724
Precision:  0.38526912181303113
Recall:  0.08018867924528301
F1:  0.13274768179599805
PR-AUC:  0.20738929722792973
Confusion Matrix:  [[13350, 217], [1560, 136]]
```

# 8th run Scores at threshold 0.5 on validation

What changed: Remove the weighted loss
Why changed: To log the results without the weighted loss
```

My own model Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.1989840003072726
Confusion Matrix:  [[13567, 0], [1696, 0]]
```

# 9th run Scores at threshold 0.5 on validation

What changed: Add the weighted loss
Why changed: To log the results with the weighted loss

```
My own model Baseline Scores
Accuracy:  0.6194719255716439
Precision:  0.16634209672184355
Recall:  0.6043632075471698
F1:  0.2608806312038687
PR-AUC:  0.20738929722792973
Confusion Matrix:  [[8430, 5137], [671, 1025]]
```

# 10th run Scores at threshold 0.5 on validation

What changed: Remove the weighted loss
Why changed: To log the results for comparison with using the WeightedRandomSampler

```
My own model Baseline Scores
Accuracy:  0.8888816091200944
Precision:  0.0
Recall:  0.0
F1:  0.0
PR-AUC:  0.1989840003072726
Confusion Matrix:  [[13567, 0], [1696, 0]]
```

# 11th run Scores at threshold 0.5 on validation

What changed: Add the WeightedRandomSampler
Why changed: To log the results for comparison with using the WeightedRandomSampler

```
My own model Baseline Scores
Accuracy:  0.6422721614361527
Precision:  0.16678470254957506
Recall:  0.5554245283018868
F1:  0.2565359477124183
PR-AUC:  0.20123723907013816
Confusion Matrix:  [[8861, 4706], [754, 942]]
```