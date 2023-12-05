import json
import fire
from collections import defaultdict
import re

def process_validset(input_file, output_prefix, src, tgt, context=None):
    data = json.load(open(input_file, 'r', encoding="utf-8"))
    if context != None:
        data = [item for item in data if item["context_type"] == context]
    with open(output_prefix+"."+src,"w", encoding="utf-8") as out_src, open(output_prefix+"."+tgt,"w", encoding="utf-8") as out_tgt,\
        open(output_prefix+".type","w", encoding="utf-8") as out_type, open(output_prefix+".suggestion","w", encoding="utf-8") as out_suggestion:
        for item in data:
            src_line = item["src"].strip()
            
            typed_seq = item["typed_seq"]
            left_context = item["left_context"]
            right_context = item["right_context"]
            target = item["target"]
            new_tgt_line = left_context + " <mask> " + right_context

            out_src.write(src_line + "\n")
            out_tgt.write(new_tgt_line + "\n")
            out_type.write(typed_seq + "\n")
            out_suggestion.write(target + "\n")
          
def filter_generated(generated):
    results = defaultdict(list)
    template = re.compile("D-(\d+)")
    for line in generated:
        if line.startswith('D-'):
            number = int(template.findall(line)[0])
            sent = line.strip().split('\t')
            if len(sent) >= 3:
                sent = sent[2]
            else:
                sent = ''
            sent = (sent + " ").replace("@@ ", "").rstrip()
            results[number].append(sent)
    return results

def extract_generation(generate_path, input_path, output_path):
    """
    generate_path: the path to fairseq-generate output
    input_path: the path to test file (json format)
    output_path: the path to save results
    """
    generated = open(generate_path, 'r', encoding="utf-8").readlines()
    test_data = json.load(open(input_path, 'r', encoding="utf-8"))
    generated_results = filter_generated(generated)
    for i in range(len(test_data)):
        test_data[i]["pred"] = generated_results[i]
    json.dump(test_data, open(output_path, 'w', encoding="utf-8"), indent=4, ensure_ascii=False)
          
            
def compute_acc(input_path, context=None):
    data = json.load(open(input_path, 'r', encoding="utf-8"))
    if context != None:
        data = [item for item in data if item["context_type"] == context]
    total = len(data)
    hit = 0
    for sample in data:
        if sample["target"] == sample["pred"][0]:
            hit += 1
    print("Acc: {:.2f}%".format(hit/total*100))

if __name__ == "__main__":
    fire.Fire()