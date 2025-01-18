# SIGformer: Sign-aware Graph Transformer for Recommendation

This is the PyTorch implementation for SIGIR 2024 paper.
> Sirui Chen, Jiawei Chen, Sheng Zhou, Bohao Wang, Shen Han, Chanfei Su, Yuqing Yuan, Can Wang 2024. SIGformer: Sign-aware Graph Transformer for Recommendation
 [arXiv link](https://arxiv.org/abs/2404.11982)

 Original code from this [repo](https://github.com/StupidThree/SIGformer).

## Datasets

| Dataset| #Users | #Items | #Interactions | Pos/Neg |
|---|---|---|---|---|
| SberZvuk | 51,267 | 46,464 | 895,266 | 1:0.22 |
| KuaiRand | 16,974 | 4,373 | 263,100 | 1:1.25 |

## Training & Evaluation
* KuaiRand
  ```bash
  python -u code/main.py --data=KuaiRand --offset=1 --alpha=0.2 --beta=1 --sample_hop=3
  ```
