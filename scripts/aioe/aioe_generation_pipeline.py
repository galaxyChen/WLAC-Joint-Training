import json
import fire
import logging
from fairseq import checkpoint_utils, utils
import time
import torch
from tqdm import tqdm
import sys
import os
project_path = os.environ["PROJECT_ROOT"]
sys.path.append(project_path)
from tasks.aioe import AIOETask
from models.aioe_model import AIOEModel
logger = logging.getLogger("Generator")

class GenerationPipeline:
    def __init__(self, model_path=None, models=None, cfg=None, task=None, use_cuda=True) -> None:
        if model_path is not None:
            logger.info("loading model(s) from {}".format(model_path))
            if type(model_path) == str:
                model_path = [model_path]
            models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(model_path)
            logger.info(cfg)
        else:
            assert models is not None and cfg is not None and task is not None
        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Optimize ensemble for generation
        for model in models:
            if model is None:
                continue
            if use_cuda:
                model.cuda()
            model.prepare_for_inference_(cfg)
            
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.use_cuda = use_cuda
        self.cfg = cfg
        self.task = task
        
    def form_batch(self, sample: dict, ids):
        sample_input = " ".join([sample["src"], "</s>", sample["left_context"], "<tip>", " ".join(sample["typed_seq"]), "<mask>", sample["right_context"]])
        sample_input = [w for w in sample_input.strip().split() if w != ""]
        sample_input = " ".join(sample_input)

        input_ids = self.preprocess(sample_input)
        input_ids = input_ids.view(-1)
        sep_idx = torch.where(input_ids == self.tgt_dict.index("</s>"))[0].item()
        src_position_ids = list(range(2, 2+sep_idx+1)) + list(range(2, 2+len(input_ids)-sep_idx-1))
        src_segment_ids = [0] * (sep_idx+1) + [1] * (len(input_ids)-sep_idx-1)
        
        input_ids = input_ids.view(1, -1)
        src_position_ids = torch.LongTensor([src_position_ids])
        src_segment_ids = torch.LongTensor([src_segment_ids])
        assert input_ids.shape[1] == src_position_ids.shape[1] == src_segment_ids.shape[1]
        src_lengths = torch.LongTensor([input_ids.shape[1]])
        ids = torch.LongTensor([ids])
        return {"ids": ids, "src_tokens": input_ids, "src_lengths": src_lengths, "src_position_ids": src_position_ids, "src_segment_ids": src_segment_ids}
    
    def preprocess(self, s):
        return self.src_dict.encode_line(s, append_eos=False, add_if_not_exist=False)
    
    def generate(self, sample: dict, ids=None, k=5):
        use_cuda = self.use_cuda
        models = self.models
        
        if ids is None:
            ids = 0
        batch = self.form_batch(sample, ids)
        src_tokens = batch["src_tokens"]
        src_lengths = batch["src_lengths"]
        src_position_ids = batch["src_position_ids"]
        src_segment_ids = batch["src_segment_ids"]
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
            src_position_ids = src_position_ids.cuda()
            src_segment_ids = src_segment_ids.cuda()
                    
        model = models[0]
        mask_logits = model.forward_encoder(
            src_tokens = src_tokens,
            src_lengths = src_lengths,
            src_position_ids = src_position_ids,
            src_segment_ids = src_segment_ids
        )
        # mask_logits: (bsz, tgt_vocab_size)
        # get top-k predictions
        topk_logits, topk_indices = torch.topk(mask_logits.view(-1), k, dim=-1)
        preds = [self.tgt_dict.symbols[i] for i in topk_indices]
        return preds

def generate(checkpoint_path, input_path, output_path, param_path=None):
    data = json.load(open(input_path, encoding="utf8"))
    pipeleine = GenerationPipeline(checkpoint_path)
    if param_path is not None:
        params = torch.load(param_path)
        pipeleine.models[0].load_state_dict(params["model"])
    for i, sample in enumerate(tqdm(data)):
        pred = pipeleine.generate(sample, ids=i)
        sample["pred"] = pred
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    fire.Fire()