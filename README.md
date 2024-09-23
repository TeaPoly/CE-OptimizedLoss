# CE-OptimizedLoss

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

- Smoothed Max Pooling Loss: Apply temporal smoothing on the logits of frames before max pooling operation. 
- MWER loss: Computes the MWER (minimum WER) Loss with beam search and negative sampling strategy.

## Q&A

> Why is the loss value negative?

The loss is just a scalar that you are trying to minimize. It's not supposed to be positive. A detailed discussion can be found [here](https://github.com/keras-team/keras/issues/1917#issuecomment-193580929).

It is normal to observe that the loss value is getting smaller and smaller, because the average word error is subtracted when normalizing. For details, please refer to the following formula from [paper](https://arxiv.org/abs/1712.01818):

![](https://user-images.githubusercontent.com/3815778/206600760-ea1927ea-e479-43b0-8aa4-14c10ede7511.png)

> What is the different between beam search and negative sampling stategy for MWER loss?

The beam search and negative sampling stategy both are method to generate multiple candidate paths.

- The negative sampling strategy to generate multiple candidate paths by randomly masking the top1 score token during the MWER training as said in [paper](https://arxiv.org/abs/2206.08317). 
- The Beam search strategy is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. 

So the negative sampling strategy is training faster than beam search strategy. MWER loss with the beam search stategy is closer to the actual calling method.


## Citations

``` bibtex
@article{gao2022paraformer,
  title={Paraformer: Fast and accurate parallel transformer for non-autoregressive end-to-end speech recognition},
  author={Gao, Zhifu and Zhang, Shiliang and McLoughlin, Ian and Yan, Zhijie},
  journal={arXiv preprint arXiv:2206.08317},
  year={2022}
}

@inproceedings{prabhavalkar2018minimum,
  title={Minimum word error rate training for attention-based sequence-to-sequence models},
  author={Prabhavalkar, Rohit and Sainath, Tara N and Wu, Yonghui and Nguyen, Patrick and Chen, Zhifeng and Chiu, Chung-Cheng and Kannan, Anjuli},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4839--4843},
  year={2018},
  organization={IEEE}
}

@misc{park2020learningdetectkeywordparts,
      title={Learning To Detect Keyword Parts And Whole By Smoothed Max Pooling}, 
      author={Hyun-Jin Park and Patrick Violette and Niranjan Subrahmanya},
      year={2020},
      eprint={2001.09246},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2001.09246}, 
}
```
