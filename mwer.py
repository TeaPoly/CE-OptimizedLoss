#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Lucky Wong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""Minimum Word Error Rate Training loss

<Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition>
    https://arxiv.org/abs/2206.08317

<Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models>
    https://arxiv.org/abs/1712.01818
"""

from typing import Optional, Tuple

import torch

from utils import (
    IGNORE_ID,
    MIN_LOG_VAL,
    make_pad_mask,
    mask_finished_preds,
    mask_finished_scores
)


def create_sampling_mask(log_softmax, n):
    """
    Generate sampling mask

    # Ref: <Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition>
    #       https://arxiv.org/abs/2206.08317

    Args:
        log_softmax: log softmax inputs, float32 (batch, maxlen_out, vocab_size)
        n: candidate paths num, int32
    Return:
        sampling_mask: the sampling mask (nbest, batch, maxlen_out, vocab_size)
    """
    b, s, v = log_softmax.size()

    # Generate random mask
    nbest_random_mask = torch.randint(
        0, 2, (n, b, s, v), device=log_softmax.device)

    # Greedy search decoding for best path
    top1_score_indices = log_softmax.argmax(dim=-1).squeeze(-1)

    # Genrate top 1 score token mask
    top1_score_indices_mask = torch.zeros((b, s, v), dtype=torch.int).to(
        log_softmax.device
    )
    top1_score_indices_mask.scatter_(-1, top1_score_indices.unsqueeze(-1), 1)

    # Genrate sampling mask by applying random mask to top 1 score token
    sampling_mask = nbest_random_mask * top1_score_indices_mask.unsqueeze(0)

    return sampling_mask


def negative_sampling_decoder(
    logit: torch.Tensor,
    nbest: int = 4,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multiple candidate paths by negative sampling strategy

    # Ref: <Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition>
    #       https://arxiv.org/abs/2206.08317

    Args:
        logit: logit inputs, float32 (batch, maxlen_out, vocab_size)
        nbest: candidate paths num, int32
        masks: logit lengths, (batch, maxlen_out)
    Return:
        nbest_log_distribution: the N-BEST distribution of candidate path (nbest, batch)
        nbest_pred: the NBEST candidate path (nbest, batch, maxlen_out)
    """

    # Using log-softmax for probability distribution
    log_softmax = torch.nn.functional.log_softmax(logit, dim=-1)

    # Generate sampling mask
    with torch.no_grad():
        sampling_mask = create_sampling_mask(log_softmax, nbest)

    # Randomly masking top1 score with -float('inf')
    # (nbest, batch, maxlen_out, vocab_size)
    nbest_log_softmax = torch.where(
        sampling_mask != 0, MIN_LOG_VAL.type_as(log_softmax), log_softmax
    )

    # Greedy search decoding for sampling log softmax
    nbest_logsoftmax, nbest_pred = nbest_log_softmax.topk(1)
    nbest_pred = nbest_pred.squeeze(-1)
    nbest_logsoftmax = nbest_logsoftmax.squeeze(-1)

    # Construct N-BEST log PDF
    # FIXME (huanglk): Ignore irrelevant probabilities
    # (n, b, s) -> (n, b): log(p1*p2*...pn) = log(p1)+log(p2)+...log(pn)
    nbest_log_distribution = torch.sum(
        nbest_logsoftmax.masked_fill(masks, 0), -1)

    return nbest_log_distribution, nbest_pred


def batch_beam_search(
    logit: torch.Tensor, beam_size: int, masks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Beam Search Decoder

    Parameters:

        logit(Tensor) – the logit of network.
        beam_size(int) – beam size of decoder.

    Outputs:

        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """
    batch_size, seq_length, vocab_size = logit.shape
    eos = vocab_size - 1
    # beam search
    with torch.no_grad():
        # b,t,v
        log_post = torch.nn.functional.log_softmax(logit, dim=-1)
        # b,k
        log_prob, indices = log_post[:, 0, :].topk(beam_size, sorted=True)
        end_flag = torch.eq(masks[:, 0], 1).view(-1, 1)
        # mask predictor and scores if end
        log_prob = mask_finished_scores(log_prob, end_flag)
        indices = mask_finished_preds(indices, end_flag, eos)
        # b,k,1
        indices = indices.unsqueeze(-1)

        for i in range(1, seq_length):
            # b,v
            scores = mask_finished_scores(log_post[:, i, :], end_flag)
            # b,v -> b,k,v
            topk_scores = scores.unsqueeze(1).repeat(1, beam_size, 1)
            # b,k,1 + b,k,v -> b,k,v
            top_k_logp = log_prob.unsqueeze(-1) + topk_scores

            # b,k,v -> b,k*v -> b,k
            log_prob, top_k_index = top_k_logp.view(batch_size, -1).topk(
                beam_size, sorted=True
            )

            index = mask_finished_preds(top_k_index, end_flag, eos)

            indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)

            end_flag = torch.eq(masks[:, i], 1).view(-1, 1)

        indices = torch.fmod(indices, vocab_size)
    return indices, log_prob


def beam_search_decoder(
    logit: torch.Tensor, beam_size: int, masks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Beam Search Decoder

    Parameters:

        logit(Tensor) – the logit of network.
        beam_size(int) – beam size of decoder.

    Outputs:

        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """
    # beam search decoder
    indices, _ = batch_beam_search(logit, beam_size, masks)
    # recompute PDF for gradient
    log_post = torch.nn.functional.log_softmax(logit, dim=-1)
    # b,t,v -> b,n,t,v
    nlog_post = log_post.unsqueeze(1).repeat(1, beam_size, 1, 1)
    # indices: b, n, t -> b, n, t
    top_k_log_post = torch.gather(
        nlog_post, -1, indices.unsqueeze(-1)).squeeze(-1)
    # b, n, t -> b, n
    topk_log_prob = torch.sum(
        top_k_log_post.masked_fill(masks.unsqueeze(1), 0), -1)
    return topk_log_prob.transpose(0, 1), indices.transpose(0, 1)


def compute_mwer_loss(
    nbest_log_distribution=torch.Tensor,
    nbest_pred=torch.Tensor,
    tgt=torch.Tensor,
    masks=torch.Tensor,
):
    """
    Compute Minimum Word Error Rate Training loss.

    # Ref: <Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models>
    #       https://arxiv.org/abs/1712.01818

    Args:
        nbest_log_distribution: the N-BEST distribution of candidate path (nbest, batch)
        nbest_pred: the NBEST candidate path (nbest, batch, maxlen_out)
        tgt: padded target token ids, int32 (batch, maxlen_out)
        masks: target token lengths of this batch (batch,)
    Return:
        loss: normalized MWER loss (batch,)
    """
    n, b, s = nbest_pred.size()

    # necessary to filter irrelevant length
    # (b,) -> (b, s)
    # not include <eos/sos>
    tgt = tgt.masked_fill(masks, IGNORE_ID)
    # (n, b, s)
    nbest_pred = nbest_pred.masked_fill(masks, IGNORE_ID)

    # Construct number of word errors
    # (b, s) -> (n, b, s)
    tgt = tgt.unsqueeze(0).repeat(n, 1, 1)

    # convert to float for normalize
    # (n, b, s) -> (n, b)
    nbest_word_err_num = torch.sum((tgt != nbest_pred), -1).float()

    # Computes log distribution
    # (n, b) -> (b,): log( p1+p2+...+pn ) = log( exp(log_p1)+exp(log_p2)+...+exp(log_pn) )
    sum_nbest_log_distribution = torch.logsumexp(nbest_log_distribution, 0)

    # Re-normalized over just the N-best hypotheses.
    # (n, b) - (b,) -> (n, b): exp(log_p)/exp(log_p_sum) = exp(log_p-log_p_sum)
    normal_nbest_distribution = torch.exp(
        nbest_log_distribution - sum_nbest_log_distribution
    )

    # Average number of word errors over the N-best hypohtheses
    # (n, b) -> (b)
    mean_word_err_num = torch.mean(nbest_word_err_num, 0)
    # print("mean_word_err_num:", mean_word_err_num)

    # Re-normalized error word number over just the N-best hypotheses
    # (n, b) - (b,) -> (n, b)
    normal_nbest_word_err_num = nbest_word_err_num - mean_word_err_num

    # Expected number of word errors over the training set.
    # (n, b) -> (b,)
    mwer_loss = torch.sum(normal_nbest_distribution *
                          normal_nbest_word_err_num, 0)

    return mwer_loss


class Seq2seqMwerLoss(torch.nn.Module):
    """Minimum Word Error Rate Training loss based on the negative sampling strategy

    <Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition>
        https://arxiv.org/abs/2206.08317

    <Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models>
        https://arxiv.org/abs/1712.01818

    Args:
        candidate_paths_num (int): The number of candidate paths.
    """

    def __init__(
        self,
        sampling_method="beam_search",  # beam_search or negative_sampling
        candidate_paths_num: int = 4,
        reduction: str = "mean",
        eos_id: int = -1,
    ):
        super().__init__()
        self.candidate_paths_num = candidate_paths_num
        self.sampling_method = sampling_method
        self.reduction = reduction
        self.eos_id = eos_id

    def forward(
        self, logit: torch.Tensor, tgt: torch.Tensor, tgt_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logit: logit (batch, maxlen_out, vocab_size)
            tgt: padded target token ids, int64 (batch, maxlen_out)
            tgt_lens: target lengths of this batch (batch)
        Return:
            loss: normalized MWER loss
        """
        assert tgt_lens.size()[0] == tgt.size()[0] == logit.size()[0]
        assert logit.size()[1] == tgt.size()[1]

        # not include <eos/sos>
        masks = make_pad_mask(
            tgt_lens if self.eos_id < 0 else tgt_lens - 1, max_len=tgt.size()[1]
        )
        if self.sampling_method == "beam_search":
            # Beam search to generate multiple candidate paths
            nbest_log_distribution, nbest_pred = beam_search_decoder(
                logit, self.candidate_paths_num, masks
            )
        elif self.sampling_method == "negative_sampling":
            # Randomly mask the top1 score to generate multiple candidate paths
            nbest_log_distribution, nbest_pred = negative_sampling_decoder(
                logit, self.candidate_paths_num, masks
            )
        else:
            raise Exception(f"Not support sampling_method: {self.sampling_method} ")

        # Compute MWER loss
        mwer_loss = compute_mwer_loss(
            nbest_log_distribution, nbest_pred, tgt, masks)

        if self.reduction == "sum":
            return torch.sum(mwer_loss)
        elif self.reduction == "mean":
            return torch.mean(mwer_loss)
        else:
            return mwer_loss


if __name__ == "__main__":
    torch.manual_seed(34)

    candidate_paths_num = 3
    vocab_size = 3
    seq_len = 4

    batch_size = 2
    lens_list = [seq_len / 2, seq_len]

    logit = torch.randn(batch_size, seq_len, vocab_size)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt_lens = torch.Tensor(lens_list).int()

    sampling_method = "beam_search"
    mwer_loss = Seq2seqMwerLoss(sampling_method, candidate_paths_num)
    loss = mwer_loss(logit, tgt, tgt_lens)
    print(loss)
