#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2024 Lucky Wong
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

"""Smoothed Max Pooling Loss

<LEARNING TO DETECT KEYWORD PARTS AND WHOLE BY SMOOTHED MAX POOLING>
    http://arxiv.org/abs/2001.09246

"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def truncated_gaussian_in_full_window(
    full_window_size=60, trunc_window_size=21, sigma=9
):
    """
    For the decoder SMP(smoothed max pooling) loss,
    truncated Gaussian as the smoothing filter s(t) with µ = 0, σ = 9 frames (90ms) 
    and truncated length 21 frames.
    Max pooling window of size 60 frames (600ms)

    Creates a 1D Gaussian filter where the center 'trunc_window_size' frames are filled
    with a truncated Gaussian, and the rest are zeros.

    Args:
    - full_window_size (int): The size of the full window over which the filter is applied (60 frames).
    - trunc_window_size (int): The truncated length to be applied to the center of the filter (21 frames).
    - sigma (float): The standard deviation of the Gaussian (9 frames).

    Returns:
    - smoothing_filter (Tensor): A 1D filter tensor of shape (1, 1, full_window_size)
                                 with the central 'trunc_window_size' frames filled by a Gaussian.
    """
    # Create a range for the truncated Gaussian part (center 21 frames)
    trunc_half = trunc_window_size // 2
    x = torch.arange(-trunc_half, trunc_half + 1, dtype=torch.float32)

    # Compute the Gaussian function for each value of x (21 frames)
    gaussian_filter = torch.exp(-0.5 * (x / sigma) ** 2)

    # Normalize the Gaussian filter to sum to 1
    gaussian_filter /= gaussian_filter.sum()

    # Create a full window of zeros (60 frames)
    full_filter = torch.zeros(full_window_size, dtype=torch.float32)

    # Find the center index of the full window
    center_idx = full_window_size // 2

    # Place the truncated Gaussian filter in the center of the full window
    full_filter[center_idx - trunc_half: center_idx +
                trunc_half + 1] = gaussian_filter

    # Reshape the filter to fit a 1D convolution: (out_channels, in_channels, full_window_size)
    smoothing_filter = full_filter.view(1, 1, -1)

    return smoothing_filter


class DecoderSmoothedMaxPoolingLoss(nn.Module):
    """
    For the decoder SMP(smoothed max pooling) loss,
    we used truncated Gaussian as the smoothing filter s(t) with µ = 0, σ = 9 frames (90ms) 
    and truncated length 21 frames.
    Max pooling window of size 60 frames (600ms) with offsetD = 40 frames (400ms) is used.
    """

    def __init__(
        self,
        win_size: int = 60,  # 600ms
        offset_d: int = 40,  # 400ms
        trunc_window_size: int = 21,  # 210ms
        sigma: int = 9,  # 90ms
    ) -> None:
        """
        Args:
        - win_size: Window size for the max pooling.
        - offset_d: Offset size.
        - trunc_window_size: Truncated window size.
        - sigma: Gaussian sigma.
        """
        super(DecoderSmoothedMaxPoolingLoss, self).__init__()
        self.win_size = win_size
        self.smoothing_filter = truncated_gaussian_in_full_window(
            full_window_size=win_size, trunc_window_size=trunc_window_size, sigma=sigma
        )
        self.offset_d = offset_d

    def forward(
        self,
        X: torch.Tensor,
        lengths: torch.Tensor,
        tgt: torch.Tensor,
        w_end: List[int],
    ):
        """
        Args:
        - X: Tensor of shape (batch_size, frames, num_class), the input sigmoid.
        - lengths: Tensor of shape (batch_size,), encoder lengths.
        - tgt: Tensor of shape (batch_size,), ground truth labels.
        - w_end: List of word end frame.

        Returns:
        - loss: Scalar tensor representing the smoothed max pooling loss.
        """
        num_utts, frames, num_keywords = X.shape

        loss = 0.0
        for i in range(num_utts):
            cur_len = lengths[i]
            part_prob = X[i, :cur_len, :]
            for j in range(num_keywords):
                prob = part_prob[:, j]
                if tgt[i] == j:
                    assert w_end[i] > 0, (w_end[i], cur_len)
                    tau_d_start = max(
                        0, w_end[i] + self.offset_d - self.win_size)
                    tau_d_end = min(tau_d_start + self.win_size, cur_len)
                    assert tau_d_start < tau_d_end, (tau_d_start, tau_d_end)

                    # Apply the smoothing filter
                    # Positive loss: log max pooling over window [tau_d_start, tau_d_end]
                    # (minibatch,in_channels, iW)
                    prob_win = prob[tau_d_start:tau_d_end].view(1, 1, -1)

                    smoothed_prob_win = F.conv1d(
                        prob_win,
                        self.smoothing_filter.to(prob_win.device),
                        padding="same",
                    ).view(-1)

                    smoothed_prob_win = torch.clamp(
                        smoothed_prob_win, 1e-8, 1.0)

                    # Find the frame with the maximum logit for each class
                    max_prob = smoothed_prob_win.max()

                    # Calculate cross entropy loss for positive frames
                    loss += -torch.log(max_prob)

                    other_prob = torch.cat(
                        (prob[:tau_d_start], prob[tau_d_end:]), 0)
                else:
                    other_prob = prob

                neg_prob = torch.clamp(1 - other_prob, 1e-8, 1.0)
                loss += - torch.log(neg_prob).sum()

        return loss
