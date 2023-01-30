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

import torch


IGNORE_ID = -1

MIN_LOG_VAL = torch.tensor(-float("inf"))


def make_pad_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = int(lengths.size(0))
    if max_len is None:
        max_len = int(lengths.max().item())
    seq_range = torch.arange(
        0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def mask_finished_preds(
    pred: torch.Tensor, flag: torch.Tensor, eos: int
) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)


def mask_finished_scores(score: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat(
            (zero_mask, flag.repeat([1, beam_size - 1])), dim=1)
        finished = torch.cat(
            (flag, zero_mask.repeat([1, beam_size - 1])), dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float("inf"))
    score.masked_fill_(finished, 0)
    return score
