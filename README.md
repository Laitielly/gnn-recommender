# SIGformer: Sign-aware Graph Transformer for Recommendation

This is the PyTorch implementation for SIGIR 2024 paper.
> Sirui Chen, Jiawei Chen, Sheng Zhou, Bohao Wang, Shen Han, Chanfei Su, Yuqing Yuan, Can Wang 2024. SIGformer: Sign-aware Graph Transformer for Recommendation
 [arXiv link](https://arxiv.org/abs/2404.11982)

 Original code from this [repo](https://github.com/StupidThree/SIGformer).

## Datasets

| Dataset| #Users | #Items | #Interactions | Pos/Neg |
|---|---|---|---|---|
| SberZvuk | 999 | 7401 | 6 911 972  | 1:1.5 |
| KuaiRand | 16,974 | 4,373 | 263,100 | 1:1.25 |

## Training & Evaluation
* KuaiRand
  ```bash
  python -u code/main.py --data=KuaiRand --offset=1 --alpha=0.2 --beta=1 --sample_hop=3
  ```
* SberZvuk (data download and preparing is required)
  ```bash
  python -u baseline/main.py --data=1kSberData --offset=0.5 --alpha=0.2 --beta=1 --sample_hop=1 --hidden_dim 8 --n_layers 1 --test_batch_size 256
  ```
## Results on test

| Dataset| Recall@20 | Precision@20 | NDCG@20 |
|---|---|---|---|
| KuaiRand | 0.15 | 0.015 | 0.072  |
| SberZvuk | 0.0014 | 0.02 | 0.023 |
