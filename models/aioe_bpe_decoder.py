from fairseq.models.transformer import TransformerDecoder
from copy import deepcopy

class AIOEBPEDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        if not self.cfg.no_ar_task:
            self.build_mt_decoder(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        else:
            print("no ar task, skipping mt decoder")


    def build_mt_decoder(self, args, dictionary, embed_tokens, no_encoder_attn, output_projection):
        mt_args = deepcopy(args)
        print("mt args", mt_args)
        mt_args.decoder.layers = args.num_mt_decoder_layers
        self.mt_decoder = TransformerDecoder(mt_args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        # share output projections
        self.output_projection = self.mt_decoder.output_projection

    def freeze_mt_parameters(self):
        for name, param in self.mt_decoder.named_parameters():
            if "embed_tokens" not in name and "output_projection" not in name:
                param.requires_grad = False