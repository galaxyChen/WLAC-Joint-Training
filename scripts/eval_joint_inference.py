import fire
import json
from collections import defaultdict

def joint_inference(input_path):
    data = json.load(open(input_path, encoding="utf8"))

    for sample in data:
        mt_hypos = sample["translation"] # a list of strings
        predictions = sample["pred"] # a list of words
        counter = defaultdict(lambda : 0)
        for word in predictions:
            for hypo in mt_hypos:
                if word in hypo:
                    counter[word] += 1
        sample["joint_inference"] = [w[0] for w in sorted(counter.items(), key=lambda x: x[1], reverse=True)]
    return data


def compute_acc(input_path):
    data = json.load(open(input_path, encoding="utf8"))
    total = len(data)
    hit = 0
    for sample in data:
        pred = sample["joint_inference"][0]
        gold = sample["target"]
        if pred == gold:
            hit += 1
    return hit / total

def eval(input_path, output_path):
    """
    input_path: The path to test file (json format). The input file should contain raw predictions (a list of predictions) and the translation hypothesis (a list of strings).
    output_path: The path to save preditions (json format)
    """
    results = joint_inference(input_path)
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    acc = compute_acc(output_path)
    print("Accuracy: ", acc)

if __name__ == '__main__':
    fire.Fire()