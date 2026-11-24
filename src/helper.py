

def print_numbers(results):
    print("Trivial Baseline Scores")
    print("Accuracy: ", results["trivial"]["accuracy"])
    print("Precision: ", results["trivial"]["precision"])
    print("Recall: ", results["trivial"]["recall"])
    print("F1: ", results["trivial"]["f1"])
    print("PR-AUC: ", results["trivial"]["pr_auc"])
    print("Confusion Matrix: ", results["trivial"]["confusion"])
    print("HGB Baseline Scores")
    print("Accuracy: ", results["hgb"]["accuracy"])
    print("Precision: ", results["hgb"]["precision"])
    print("Recall: ", results["hgb"]["recall"])
    print("F1: ", results["hgb"]["f1"])
    print("PR-AUC: ", results["hgb"]["pr_auc"])
    print("Confusion Matrix: ", results["hgb"]["confusion"])
    print("My own model Baseline Scores")
    print("Accuracy: ", results["model"]["accuracy"])
    print("Precision: ", results["model"]["precision"])
    print("Recall: ", results["model"]["recall"])
    print("F1: ", results["model"]["f1"])
    print("PR-AUC: ", results["model"]["pr_auc"])
    print("Confusion Matrix: ", results["model"]["confusion"])


def plot_history(history):
    ...
