

def print_numbers(results):
    names = [("trivial", "Trivial"), ("hgb", "HGB"), ("model", "My model")]
    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("pr_auc", "PR-AUC"),
    ]

    # full precision: repr(float) is the exact shortest form, nothing is rounded
    cells = {
        key: [repr(float(results[name][key])) for name, _ in names]
        for key, _ in metrics
    }

    label_w = max(len(label) for _, label in metrics) + 2
    col_ws = [
        max([len(label)] + [len(cells[key][i]) for key, _ in metrics]) + 2
        for i, (_, label) in enumerate(names)
    ]

    header = "Metric".ljust(label_w) + "".join(
        label.rjust(w) for (_, label), w in zip(names, col_ws)
    )
    line = "-" * len(header)

    print(line)
    print(header)
    print(line)

    for key, label in metrics:
        row = label.ljust(label_w) + "".join(
            cell.rjust(w) for cell, w in zip(cells[key], col_ws)
        )
        print(row)

    print(line)

    print("\nConfusion matrices (rows = actual, columns = predicted)")

    for name, label in names:
        (tn, fp), (fn, tp) = results[name]["confusion"]

        print(f"\n{label}")
        print(f"{'':>10}{'pred 0':>10}{'pred 1':>10}")
        print(f"{'actual 0':>10}{tn:>10}{fp:>10}")
        print(f"{'actual 1':>10}{fn:>10}{tp:>10}")


def plot_history(history):
    ...


def plot_history(history):
    ...
