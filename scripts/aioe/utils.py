import json
import fire

def process_validset(input_file, output_prefix, src, tgt, context=None):
    data = json.load(open(input_file, 'r'))
    if context != None:
        data = [item for item in data if item["context_type"] == context]
    with open(output_prefix+"."+src,"w") as out_src, open(output_prefix+"."+tgt,"w") as out_tgt,\
        open(output_prefix+".type","w") as out_type, open(output_prefix+".suggestion","w") as out_suggestion:
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
            
def compute_acc(input_path, context=None):
    data = json.load(open(input_path, 'r'))
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