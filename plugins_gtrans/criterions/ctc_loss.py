# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from math import log

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor
import numpy as np
from ..modules.nonautoregressive_gtransformer import tokens2tags

def tags2index(tags):
    tag_max = tags.max()
    index_tag = torch.arange(1, tag_max + 1, device=tags.device)
    mask_tag = tags.unsqueeze(1) == index_tag.view(1, -1, 1)
    ntok_tag = mask_tag.sum(-1)
    # get token indexes of each tag
    ntok_mean = ntok_tag[ntok_tag > 0].float().mean().long()
    ntok_max = ntok_tag.max()
    ntok_top = min(ntok_max, ntok_mean * 3)
    index_tok = torch.arange(0, tags.size(1), device=tags.device)
    index_pad = 99999
    index_tok = mask_tag * index_tok.view(1, 1, -1) + (~mask_tag) * index_pad
    index_tok = torch.topk(index_tok, ntok_top, dim=-1, largest=False).values
    mask_tok = index_tok != index_pad
    index_tok = mask_tok * index_tok
    return index_tok, mask_tok

def index2value(input, index, mask, padding):
    index0 = torch.arange(0, index.size(0)).view(-1, 1, 1)
    value = input[index0, index]
    if len(value.size()) > len(mask.size()):
        mask = mask.unsqueeze(-1)
    return value * mask + padding * (~mask)

def tokens2sents(tokens, lprobs=None):
    # get start and end positions of all sentences
    sents = []
    for line in tokens.tolist():
        sline = []
        for i in range(len(line)):
            if line[i] == 0:  # bos
                sline.append([i, -1])
            elif line[i] == 2:  # eos
                sline[-1][1] = i
        assert np.all(t > s for s, t in sline)
        sents.append(sline)
    # split the sentences
    nsent = np.sum(len(sline) for sline in sents)
    max_len = np.max([np.max([t - s + 1 for s, t in sline]) for sline in sents])
    new_tokens = tokens.new_ones((nsent, max_len))
    i = j = 0
    for sline in sents:
        for s, t in sline:
            new_tokens[j, : t - s + 1] = tokens[i, s: t + 1]
            j += 1
        i += 1
    # lprobs
    if lprobs is not None:
        new_lprobs = lprobs.new_zeros((nsent, max_len, lprobs.size(-1)))
        i = j = 0
        for sline in sents:
            for s, t in sline:
                new_lprobs[j, : t - s + 1, :] = lprobs[i, s: t + 1, :]
                j += 1
            i += 1
        return new_tokens, new_lprobs
    else:
        return new_tokens

@register_criterion("ctc_loss")
class LabelSmoothedCTCCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.blank_id = task.tgt_dict.unk()
        self.ctc_loss = torch.nn.CTCLoss(blank=self.blank_id, reduction='none', zero_infinity=True)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument('--mse-lambda', default=10, type=float, metavar='D')

    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _compute_ctc_loss(
            self, outputs, prev_output_tokens, targets, output_mask, target_mask, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        lprobs = F.log_softmax(outputs, dim=-1, dtype=torch.float32)

        # Guangsheng Bao: split sentences for CTC loss
        # prev_output_tokens, lprobs = tokens2sents(prev_output_tokens, lprobs=lprobs)
        # targets = tokens2sents(targets)
        # output_mask = prev_output_tokens.ne(self.padding_idx)
        # target_mask = targets.ne(self.padding_idx)

        # Guangsheng Bao: split sentences for CTC loss
        tags_prev = tokens2tags(self.task.target_dictionary, prev_output_tokens)
        tags_tgt = tokens2tags(self.task.target_dictionary, targets)
        index_prev, mask_prev = tags2index(tags_prev)
        index_tgt, mask_tgt = tags2index(tags_tgt)
        lprobs = index2value(lprobs, index_prev, mask_prev, 0.0)
        targets = index2value(targets, index_tgt, mask_tgt, self.padding_idx)
        lprobs = lprobs.view(-1, lprobs.size(2), lprobs.size(3))
        mask_prev = mask_prev.view(-1, mask_prev.size(2))
        targets = targets.view(-1, targets.size(2))
        mask_tgt = mask_tgt.view(-1, mask_tgt.size(2))
        # filter empty tags
        mask_item = mask_tgt.any(-1)
        lprobs = lprobs[mask_item]
        targets = targets[mask_item]
        output_mask = mask_prev[mask_item]
        target_mask = mask_tgt[mask_item]

        # output_mask = mask_prev
        # target_mask = mask_tgt
        output_lens = output_mask.sum(-1)
        target_lens = target_mask.sum(-1)
        losses = self.ctc_loss(lprobs.transpose(0, 1), targets, output_lens, target_lens)
        ctc_loss = losses.sum()/(output_lens.sum().float())
        lprobs = lprobs[output_mask]
        if label_smoothing > 0:
            loss = (
                    ctc_loss * (1 - label_smoothing) - mean_ds(lprobs) * label_smoothing
            )
        else:
            loss = ctc_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "ctc_loss": ctc_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]
        if 'glat' in sample:
            glat = sample['glat']
        else:
            glat = None
        prev_output_tokens = model.initialize_ctc_input(src_tokens)
        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat)
        losses, ctc_loss = [], []

        for obj in outputs:
            if obj.startswith('glat'):
                continue
            if obj == "word_ins":
                _losses = self._compute_ctc_loss(
                    outputs[obj].get("out"),
                    prev_output_tokens,
                    outputs[obj].get("tgt"),
                    prev_output_tokens.ne(self.task.tgt_dict.pad()),
                    tgt_tokens.ne(self.task.tgt_dict.pad()),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            elif outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("ctc_loss", False):
                ctc_loss += [_losses.get("ctc_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        ctc_loss = sum(l for l in ctc_loss) if len(ctc_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "ctc_loss": ctc_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if "glat_accu" in outputs:
            logging_output["glat_accu"] = outputs['glat_accu']
        if "glat_context_p" in outputs:
            logging_output['glat_context_p'] = outputs['glat_context_p']

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        log_metric("glat_accu", logging_outputs)
        log_metric("glat_context_p", logging_outputs)

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


def log_metric(key, logging_outputs):
    if len(logging_outputs) > 0 and key in logging_outputs[0]:
        metrics.log_scalar(
            key, utils.item(np.mean([log.get(key, 0) for log in logging_outputs])), priority=10, round=3
        )