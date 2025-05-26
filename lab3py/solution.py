import csv
import sys
from typing import Optional
from collections import Counter, defaultdict
import math

def parse_args():
    if len(sys.argv) < 3:
        print("Usage: python solution.py <train_csv> <test_csv> [<max_depth>]")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else None

    return train_path, test_path, max_depth

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows

class ID3:
    def __init__(self, max_depth: Optional[int] = None):
        self.tree = None
        self.target_name = None
        self.max_depth = max_depth

    def fit(self, dataset):
        self.target_name = dataset[0][-1]
        header = dataset[0]
        data = dataset[1:]

        if self.max_depth == 0:
            self.tree = self._majority_class(data)
            print("[BRANCHES]:")
            print(self.tree)
            return

        self.tree = self._build_tree(data, header[:-1], 1)
        print("[BRANCHES]:")
        self._print_tree(self.tree)

    def _majority_class(self, data):
        counts = Counter(row[-1] for row in data)
        max_count = max(counts.values())
        majority = sorted([val for val, cnt in counts.items() if cnt == max_count])[0]
        return majority

    def _build_tree(self, data, attributes, depth):
        target_vals = [row[-1] for row in data]

        if all(val == target_vals[0] for val in target_vals):
            return target_vals[0]

        if not attributes:
            return self._majority_class(data)

        if self.max_depth is not None and depth > self.max_depth:
            return self._majority_class(data)

        gains = [(attr, self._info_gain(data, i)) for i, attr in enumerate(attributes)]
        gains.sort(key=lambda x: (-x[1], x[0]))
        print(" ".join([f"IG({attr})={gain:.4f}" for attr, gain in gains]))

        best_attr = gains[0][0]
        best_index = attributes.index(best_attr)

        tree = {"attribute": best_attr, "depth": depth, "branches": {}}
        attr_values = set(row[best_index] for row in data)

        for val in sorted(attr_values):
            subset = [row[:best_index] + row[best_index+1:] for row in data if row[best_index] == val]
            subtree = self._build_tree(
                subset,
                attributes[:best_index] + attributes[best_index+1:],
                depth + 1
            )
            tree["branches"][val] = subtree

        tree["majority"] = self._majority_class(data)
        return tree

    def _info_gain(self, data, attr_index):
        total_entropy = self._entropy(data)
        total = len(data)
        partitions = defaultdict(list)
        for row in data:
            partitions[row[attr_index]].append(row)
        weighted_entropy = sum((len(part)/total) * self._entropy(part) for part in partitions.values())
        return total_entropy - weighted_entropy

    def _entropy(self, data):
        target_index = -1
        counts = Counter(row[target_index] for row in data)
        total = len(data)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())

    def _print_tree(self, node, path=[]):
        if isinstance(node, str):
            print(" ".join(path), node)
            return
        attr = node["attribute"]
        for val, subtree in node["branches"].items():
            branch = f"{node['depth']}:{attr}={val}"
            self._print_tree(subtree, path + [branch])

    def predict(self, dataset):
        header = dataset[0]
        data = dataset[1:]
        predictions = [self._predict_single(row, self.tree, header) for row in data]
        print("[PREDICTIONS]:", " ".join(predictions))

        actuals = [row[-1] for row in data]
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        accuracy = correct / len(actuals)
        print(f"[ACCURACY]: {accuracy:.5f}")

        labels = sorted(set(actuals + predictions))
        matrix = {label: {l: 0 for l in labels} for label in labels}
        for a, p in zip(actuals, predictions):
            matrix[a][p] += 1
        print("[CONFUSION_MATRIX]:")
        for a in labels:
            print(" ".join(str(matrix[a][p]) for p in labels))

        return predictions

    def _predict_single(self, row, tree, header):
        node = tree
        while isinstance(node, dict):
            attr = node["attribute"]
            attr_idx = header.index(attr)
            val = row[attr_idx]
            if val in node["branches"]:
                node = node["branches"][val]
            else:
                return node["majority"]
        return node

def main():
    train_path, test_path, max_depth = parse_args()

    train_header, train_data = load_dataset(train_path)
    test_header, test_data = load_dataset(test_path)

    train_dataset = [train_header] + train_data
    test_dataset = [test_header] + test_data

    model = ID3(max_depth=max_depth)
    model.fit(train_dataset)
    model.predict(test_dataset)

if __name__ == "__main__":
    main()
