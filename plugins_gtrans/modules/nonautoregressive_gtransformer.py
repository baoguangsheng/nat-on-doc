# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel, ensemble_decoder, ensemble_encoder
from .gtransformer import GTransformerEncoder, GTransformerDecoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t

# Guangsheng Bao: generate group-tags according token sequence
def tokens2tags(dict, tokens):
    mask_tag = ~(tokens == dict.pad_index)
    start_tag = (tokens == dict.bos_index).int()
    tok_tags = torch.cumsum(start_tag, dim=1) * mask_tag.int()
    return tok_tags

@register_model("nonautoregressive_gtransformer")
class NAGTransformerModel(FairseqNATModel):
    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        encoder = NAGTransformerEncoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NAGTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt, length_mask = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "mask": length_mask,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        length_tgt, length_mask = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )
        if len(length_tgt.size()) == 0:
            length_tgt = length_tgt.unsqueeze(0)

        # Guangsheng Bao: predict the length for each sentence
        length_tgt = length_tgt.clamp_(min=2) * length_mask
        length_all = length_tgt.sum(-1)
        max_length = length_all.max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_all[:, None], self.unk
        )
        # Guangsheng Bao: set the bos and eos for each sentence
        start_tgt = torch.cat([length_tgt[:, -1:] * 0, length_tgt[:, :-1]], dim=-1) * length_mask
        start_tgt = torch.cumsum(start_tgt, dim=-1)
        end_tgt = torch.cumsum(length_tgt, dim=-1) - 1
        initial_output_tokens.scatter_(1, start_tgt, self.bos)
        initial_output_tokens.scatter_(1, end_tgt, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )


class NAGTransformerEncoder(GTransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.ensemble_models = None
        # self.len_embed = torch.nn.Parameter(torch.Tensor(1, args.encoder_embed_dim))
        # torch.nn.init.normal_(self.len_embed, mean=0, std=0.02)

    @ensemble_encoder
    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        x, encoder_embedding, src_tags = self.forward_embedding(src_tokens, token_embeddings)

        encoder_pos = self.embed_positions(src_tokens)
        # account for padding while computing the representation
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # Guangsheng Bao: the partial mode enables the Group Attention (code name: local_attn)
        # Here we generate attention mask for Group Attention according to group tags
        local_attn_mask = None
        if self.partial_mode:
            local_attn_mask = src_tags.unsqueeze(1) != src_tags.unsqueeze(2)
            local_attn_mask &= 0 != src_tags.unsqueeze(2)

        # x = torch.cat([self.len_embed.unsqueeze(0).repeat(src_tokens.size(0),1,1), x], dim=1)
        x = self.dropout_module(x)
        # if encoder_padding_mask is not None:
        #     encoder_padding_mask = torch.cat(
        #         [encoder_padding_mask.new(src_tokens.size(0), 1).fill_(0), encoder_padding_mask], dim=1)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)
        # encoder layers
        for layer in self.layers:
            x, _ = layer(
                x, padding_mask=encoder_padding_mask if has_pads else None, local_attn_mask=local_attn_mask
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        len_x = x[0, :, :]
        # x = x[1:, :, :]
        # if encoder_padding_mask is not None:
        #     encoder_padding_mask = encoder_padding_mask[:, 1:]
        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_tags": src_tags,  # B x T
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_pos": [encoder_pos],
            "length_out": [len_x],
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # tags for each token
        src_tags = tokens2tags(self.dictionary, src_tokens)
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        # x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed, src_tags

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_tags"]) == 0:
            new_encoder_tags = []
        else:
            new_encoder_tags = [encoder_out["encoder_tags"][0].index_select(1, new_order)]
        if len(encoder_out["length_out"]) == 0:
            new_length_out = []
        else:
            new_length_out = [encoder_out["length_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_pos"]) == 0:
            new_encoder_pos = []
        else:
            new_encoder_pos = [
                encoder_out["encoder_pos"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_tags": new_encoder_tags,  # B x T
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_pos": new_encoder_pos,
            "length_out": new_length_out,
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

class NAGTransformerDecoder(GTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.ensemble_models = None
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(getattr(args, "max_target_positions", 256), self.encoder_embed_dim, None)
        torch.nn.init.normal_(self.embed_length.weight, mean=0, std=0.02)
        if self.src_embedding_copy:
            self.copy_attn = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        # enc_feats = encoder_out["length_out"][0].squeeze(0)  # T x B x C
        # if len(encoder_out["encoder_padding_mask"]) > 0:
        #     src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        # else:
        #     src_masks = None
        # enc_feats = _mean_pooling(enc_feats, src_masks)
        # Guangsheng Bao: predict length for each sentence
        enc_feats = encoder_out["encoder_out"][0]
        encoder_tags = encoder_out['encoder_tags']
        max_tag = encoder_tags.max()
        all_tags = torch.arange(1, max_tag + 1, device=encoder_tags.device)
        mask_tags = encoder_tags.unsqueeze(1) == all_tags.view(1, -1, 1)
        cnt_tags = mask_tags.sum(-1)
        feat_tags = (mask_tags.unsqueeze(-1) * enc_feats.transpose(0,1).unsqueeze(1)).sum(2)
        epsilon = torch.tensor(1e-6, dtype=feat_tags.dtype, device=feat_tags.device)
        enc_feats = feat_tags / (cnt_tags.unsqueeze(-1) + epsilon)

        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)

        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embed tags
        decoder_tags = None
        if self.self_partial_mode or self.cross_partial_mode:
            prev_output_tags = tokens2tags(self.dictionary, prev_output_tokens)
            decoder_tags = prev_output_tags

        # Guangsheng Bao: the partial mode enables the Group Attention (code name: local_attn)
        # Here we generate attention mask for Group Attention and Global Attention according to group tags
        encoder_local_mask = None
        encoder_copy_mask = None
        local_attn_mask = None
        global_attn_mask = None
        # cross attention
        if self.cross_partial_mode:
            encoder_tags = encoder_out['encoder_tags'].unsqueeze(1)
            encoder_local_mask = encoder_tags != decoder_tags.unsqueeze(2)
            encoder_local_mask &= 0 != decoder_tags.unsqueeze(2)
            encoder_copy_mask = encoder_local_mask
        # self attention
        if self.self_partial_mode:
            local_attn_mask = prev_output_tags.unsqueeze(1) != decoder_tags.unsqueeze(2)
            local_attn_mask &= 0 != decoder_tags.unsqueeze(2)

        assert prev_output_tokens.size(1) < self.embed_positions.max_positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )
        # embedding
        if embedding_copy:
            src_embd = encoder_out["encoder_embedding"][0]
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None

            bsz, seq_len = prev_output_tokens.size()
            attn_score = torch.bmm(self.copy_attn(positions),
                                   (src_embd + encoder_out['encoder_pos'][0]).transpose(1, 2))
            if src_mask is not None:
                attn_score = attn_score.masked_fill(src_mask.unsqueeze(1).expand(-1, seq_len, -1), float('-inf'))
            # Guangsheng Bao: mask copy-attention by group-tag
            if encoder_copy_mask is not None:
                attn_score = attn_score.masked_fill(encoder_copy_mask, float('-inf'))
            attn_weight = F.softmax(attn_score, dim=-1)
            x = torch.bmm(attn_weight, src_embd)
            mask_target_x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)
            output_mask = prev_output_tokens.eq(self.unk)
            cat_x = torch.cat([mask_target_x.unsqueeze(2), x.unsqueeze(2)], dim=2).view(-1, x.size(2))
            sel_idx = torch.arange(bsz * seq_len).cuda() * 2 + output_mask.view(-1).long()
            assert cat_x.size(0) > sel_idx.max()
            x = cat_x.index_select(dim=0, index=sel_idx).reshape(bsz, seq_len, x.size(2))
        else:
            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        positions = positions.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            if positions is not None and (i==0 or embedding_copy):
                x += positions
                x = self.dropout_module(x)

            encoder_padding_mask = encoder_out["encoder_padding_mask"][0] \
                if (encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0) else None

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_local_mask,
                encoder_padding_mask,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # positions = (
        #     self.embed_positions(prev_output_tokens)
        #     if self.embed_positions is not None
        #     else None
        # )
        # embed tokens and positions
        if states is None:
            x = self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        # if positions is not None:
        #     x += positions
        # x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        # enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        # if len(encoder_out["encoder_padding_mask"]) > 0:
        #     src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        # else:
        #     src_masks = None
        # Guangsheng Bao: predict length for each sentence
        encoder_tags = encoder_out['encoder_tags']
        max_tag = encoder_tags.max()
        all_tags = torch.arange(1, max_tag + 1, device=encoder_tags.device)
        mask_tags = encoder_tags.unsqueeze(1) == all_tags.view(1, -1, 1)
        src_lengs = mask_tags.sum(-1)
        src_lengs = src_lengs.long()
        length_mask = (src_lengs > 0)

        if tgt_tokens is not None:
            # obtain the length target
            # tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            # Guangsheng Bao: predict length for each sentence
            tgt_tags = tokens2tags(self.dictionary, tgt_tokens)
            mask_tags = tgt_tags.unsqueeze(1) == all_tags.view(1, -1, 1)
            tgt_lengs = mask_tags.sum(-1)
            tgt_lengs = tgt_lengs.long()
            if self.pred_length_offset:
                length_tgt = (tgt_lengs - src_lengs + 128) * length_mask
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=1023)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = (pred_lengs - 128 + src_lengs) * length_mask
            else:
                length_tgt = pred_lengs * length_mask
            length_tgt = length_tgt.clamp(min=0, max=1023)

        return length_tgt, length_mask

@register_model_architecture(
    "nonautoregressive_gtransformer", "nonautoregressive_gtransformer"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.layer_norm = getattr(args, "layer_norm", "normal")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


