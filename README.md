# Data Mining Lab

- Director [Je-Hyuk Lee](https://github.com/jaylee07)

  - Assistant Professor @ Dept of AI, Bigdata & Management, Kookmin University

</br>

## RecSys

### ProLogue

- **`2024.01.18.` What? RecSys**

</br>

### Collaborative Filtering based RecSys

- **`2024.02.01.` Memory based Collaborative Filtering**
  - **Memory based Collaborative Filtering** </br> ["Using Collaborative Filtering to Weave an Information Tapestry", Goldberg et al., ACM, 1992](https://dl.acm.org/doi/abs/10.1145/138859.138867)

- **`2024.02.22.` Classical Latent Factor Model**
  - **Matrix Factorization** </br> ["Matrix Factorization Techniques for Recommender Systems", Koren et al., IEEE, 2009](https://ieeexplore.ieee.org/abstract/document/5197422?casa_token=MegLN5OlT4oAAAAA:gNQRE3BKlHAKav64qSELmwXR6WizC4ksr3XvAV1DmiLN2AFgy-PdZ9PB8gCIsgS2e1ISZNI2Oibs)

- **`2024.03.06.` Latent Factor Model for Ranking Prediction**
  - **BPR** </br> ["BPR: Bayesian Personalized Ranking from Implicit Feedback", Rendle et al., UAI, 2009](https://arxiv.org/abs/1205.2618)

- **`2024.03.20.` User Free Model**
  - **SLIM** </br> ["SLIM: Sparse Linear Methods for Top-N Recommender Systems", Ning and Karypis, ICDM, 2011](https://ieeexplore.ieee.org/abstract/document/6137254?casa_token=hasquFQkcNQAAAAA:ahz0llpC6_q77EiwLrjlyofGfms6lQOCmuBRrnGl8MOjkbLsWNWRHYJJN9yYBdXkaLKKTvNpjLiC)

  - **FISM** </br> ["FISM: Factored Item Similarity Models for Top-N Recommender Systems", Kabbur et al., KDD, 2013](https://dl.acm.org/doi/abs/10.1145/2487575.2487589?casa_token=zDZvzz_byroAAAAA:1Dr1GXJ7yst1AM9GKAlEyDRP6_hzDEQQr5ML9cjR7u6bJOr4dOp4gA3RyLyI-tVdsewY6FL7Sixq4Vs)

</br>

### Latent Factor Model leveraging Deep Learning Techs

- **`2024.04.03.` NCF**
  - [**Neural Collaborative Filtering**](https://github.com/jayarnim/MD-Data_Mining_Lab/blob/main/model/NCF.py) </br> ["Neural Collaborative Filtering", He et al., WWW, 2017](https://dl.acm.org/doi/abs/10.1145/3038912.3052569?casa_token=xJcQ62dMU8kAAAAA:erA0iE1l2Pxdx8qpbMFCh7Z6-qc02h-yCXcoaWJN5E4pJwMwu6RVRoMrBdUSFJ_yrHGdTfVtJR67EPw)

- **`2024.04.17.` AutoEncoder based Latent Factor Model**
  - **AutoRec** </br> ["AutoRec: Autoencoders Meet Collaborative Filtering", Sedhain et al., WWW, 2015](https://dl.acm.org/doi/abs/10.1145/2740908.2742726?casa_token=h6-W8fBHMuwAAAAA:hcZXeeqXUng_hrZJ9GaPt3dfJ4lXKK_THtypbucIf-XV18hRNfMxj2CkZKTOShkdwCCcrJ5WEGho-mo)

  - [**Deep AutoRec**](https://github.com/jayarnim/M-Data_Mining_Lab/blob/main/model/DeepAutoRec.py) </br> [“Training Deep AutoEncoders for Collaborative Filtering”, Kuchaiev and Ginsburg, 2017](https://arxiv.org/abs/1708.01715)

</br>

### Hybrid Latent Factor Model

- **`2024.05.01.` FM Series**
  - [**Factorization Machine**]() </br> ["Factorization Machines", Rendle, ICDM, 2010](https://ieeexplore.ieee.org/abstract/document/5694074?casa_token=PxTxcXYbSBEAAAAA:94LVL0iDWaWBXagioWFO-JagI4rp2mGkpcl-agJtPsKwhs7WhMS-f5mitp-OrI5z8M2bcAUrzLBR)

  - [**Neural Factorization Machine**](https://github.com/jayarnim/M-Data_Mining_Lab/blob/main/model/NeuralFactorizationMachine.py) </br> ["Neural Factorization Machines for Sparse Predictive Analytics", He and Chua, SIGIR, 2017](https://dl.acm.org/doi/abs/10.1145/3077136.3080777?casa_token=GwAdLrQPwy4AAAAA:ie1lvyHs54HbZmQS4pns-P585Knu3QIYRcNXUbPbfyQdNIO-E2HGXQCIwoza5np_wt-S4gs1lcQ_yw4)

  - **[Deep Factorization Machine]()** </br> ["DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", Guo et al., IJCAI, 2017](https://arxiv.org/abs/1703.04247)

  - **Field-aware Factorization Machine** </br> ["Field-aware Factorization Machines for CTR Prediction", Juan et al., RecSys, 2016](https://dl.acm.org/doi/abs/10.1145/2959100.2959134?casa_token=LhyqvBbTAH4AAAAA:j1IOKYkeCTiByjmyaTueiRLCZkmi5U0SWqEVOyBbOdZOj9xKlu7X8AeBWPsum8IwcP6hUdTHqvJgfcM)

  - **Attentional Factorization Machine** </br> ["Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks", Xiao et al., IJCAI, 2017](https://arxiv.org/abs/1708.04617)

</br>

### RecSys with Time Information

- **`2024.05.22.` Temporal Latent Factor Model**

  - **TimeSVD++** </br> [Koren, "Collaborative Filtering with Temporal Dynamic", KDD, 2009](https://dl.acm.org/doi/abs/10.1145/1557019.1557072?casa_token=rimTEX65IIsAAAAA:jfa7Vyrl6bt4D3OsxxBP2ja1FfR6DbK7EjnoTTgxbddG16QaJzh0QTSTppGwWkaJVG0nvRMba_jmB3w)

- **`2024.06.05.` Markov Chains based Sequencial Latent Factor Model**
  - **FPMC** </br> [Rendle et al., "Factorizing personalized Markov chains for next-basket recommendation", WWW, 2010](https://dl.acm.org/doi/abs/10.1145/1772690.1772773?casa_token=Q3sHZL_spjgAAAAA:2Xm7ovGfhXZSkNb2ulgWO27DY0vMDKkoVrQS23pMKqouoJS1y_AKVeQlCMI_tCsuyggAGMY-IgYrXeU)

- **`2024.06.19.` RNN based Sequencial RecSys**
  - **GRU4REC** </br> [Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks", ICLR, 2016](https://arxiv.org/abs/1511.06939)

- **`2024.06.26.` Self Attention Mechanism based RecSys**
  - **SASREC** </br> [Kang and McAuley, "Self-Attentive Sequential Recommendation", ICDM, 2018](https://ieeexplore.ieee.org/abstract/document/8594844?casa_token=JT5smtt5Z5sAAAAA:lFfXP_q_01zzLRSEc7p1zEyR_jZ7l1VjeTTCOUO6QMkDmw6HUM0BDtBSnPGpvH6XZmxvQwnGi-r7)

- **`2024.07.03.` BERT based Sequencial RecSys**
  - **BERT4REC** </br> [Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer", CIKM, 2019](https://dl.acm.org/doi/abs/10.1145/3357384.3357895?casa_token=FdOnUIipxhwAAAAA:jXWonRcvhqi5WJFCb_hKPdJMAWgvZI9YJzI4qn20pSMM7N6FrxdvcL9g9h1pAibEFy5eiD_z4N9XmbE)

</br>

### RecSys leveraging Generative Model

- **`2024.07.10.` VAE based Latent Factor Model**
  - **Mult-VAE** </br> [Liang et al., "Variational Autoencoders for Collaborative Filtering", WWW, 2018](https://dl.acm.org/doi/abs/10.1145/3178876.3186150)

- **`2024.07.17.` GAN based Collaborative Filtering**
  - **IRGAN** </br> [Wang et al., "IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models", SIGIR, 2017](https://dl.acm.org/doi/abs/10.1145/3077136.3080786?casa_token=l3DUV8WZZPUAAAAA:gh1OnSEylDd-KiNnTyq2jTgCcIAutcHOYKgFk9rXXmzdy8t8lJjfYi0XJDVzEVIsENZs8wlTCZeN_Wc)

</br>

### Graph based RecSys

- **`2024.07.24.` PageRank based RecSys**
  - **Pixie** </br> [Eksombatchai et al., "Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time", WWW, 2018](https://dl.acm.org/doi/abs/10.1145/3178876.3186183)

- **`2024.07.31.` DeepWalk based RecSys**
  - **PROD2VEC** </br> [“Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba”, Wang et al., KDD, 2018]()

- **`2024.08.07.` GraphSAGE based RecSys**
  - **PinSAGE** </br> [Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems", KDD, 2018](https://dl.acm.org/doi/abs/10.1145/3219819.3219890?casa_token=Au-umXQUZ1kAAAAA:lJzYsga18v6bN9pxyApAxnegROTbuvoCB8ukqZ3A8NiPKxY7sfXdSHsvu4eCIWgtQFoS0AaZFSzjHHY)

  - **NGCF** </br> [Wang et al., "Neural Graph Collaborative Filtering", SIGIR, 2019](https://dl.acm.org/doi/abs/10.1145/3331184.3331267?casa_token=8JTOV4RxYlsAAAAA:bkwRnHjoNWGcx5bGw97-cRpFT4iKhBSLnEyI3xK0eXEsb2-bLIwANoE1txFvyRCsgpABkhbCzrtjRA)

- **`2024.08.14.` Knowledge Graph based RecSys**
  - **RippleNet** </br> [Wang et al., “RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems”, CIKM, 2018](https://dl.acm.org/doi/abs/10.1145/3269206.3271739?casa_token=R1-vKJgCzrsAAAAA:x-U83HRTCb83izvU4lkdL29VKSeUgBBgFOpgWmjwpsa6PGdjVig-jaoUI6YdzKY6LihmfGshjhcp2Ks)

  - **KGAT** </br> [Wang et al., "KGAT: Knowledge Graph Attention Network for Recommendation", KDD, 2019](https://dl.acm.org/doi/abs/10.1145/3292500.3330989?casa_token=H-IaOAQVwHwAAAAA:2299fELWgPC7Y7f14vmWKDt0ZhrWV3I01NYuM6s1CoOyEwrltgYDzs1jP6GK_zU6v5qiwXHByDAqmIQ)

</br>



</br>

## Reference

- 추천 시스템, 차루 아가르왈 저

- [딥러닝을 활용한 추천 시스템 구현](https://fastcampus.co.kr/data_online_rs), [이재원](https://github.com/jaewonlee-728), 패스트캠퍼스

- [30개 프로젝트로 끝내는 추천 시스템 초격차 패키지](https://fastcampus.co.kr/data_online_rsystem), 패스트캠퍼스
