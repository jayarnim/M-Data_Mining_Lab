# Data Mining Lab

- **Director Je-Hyuk Lee** [`LINKEDIN`](https://linkedin.com/in/jehyuk-lee-528354112) [`GITHUB`](https://github.com/jaylee07)

  - Assistant Professor @ Dept of AI, Bigdata & Management, Kookmin University

</br>

## RecSys

### ProLogue

#### `2024.01.18.` What? RecSys

</br>

### Collaborative Filtering based RecSys

#### `2024.02.01.` Memory based Collaborative Filtering

- **Memory based Collaborative Filtering** [`REVIEW`]()

    - **Paper** : ["Using Collaborative Filtering to Weave an Information Tapestry", Goldberg et al., ACM, 1992](https://dl.acm.org/doi/abs/10.1145/138859.138867)

#### `2024.02.22.` Classical Latent Factor Model

- **Matrix Factorization** [`REVIEW`]()

    - **Paper** : ["Matrix Factorization Techniques for Recommender Systems", Koren et al., IEEE, 2009](https://ieeexplore.ieee.org/abstract/document/5197422?casa_token=MegLN5OlT4oAAAAA:gNQRE3BKlHAKav64qSELmwXR6WizC4ksr3XvAV1DmiLN2AFgy-PdZ9PB8gCIsgS2e1ISZNI2Oibs)

#### `2024.03.06.` Latent Factor Model for Ranking Prediction

- **BPR** [`REVIEW`]()

    - **Paper** : ["BPR: Bayesian Personalized Ranking from Implicit Feedback", Rendle et al., UAI, 2009](https://arxiv.org/abs/1205.2618)
    - **DataSet** : [`Netflix Prize`](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) [`Rossman Store Sales`](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales)

#### `2024.03.20.` User Free Model

- **SLIM** [`REVIEW`]()
    - **Paper** : ["SLIM: Sparse Linear Methods for Top-N Recommender Systems", Ning and Karypis, ICDM, 2011](https://ieeexplore.ieee.org/abstract/document/6137254?casa_token=hasquFQkcNQAAAAA:ahz0llpC6_q77EiwLrjlyofGfms6lQOCmuBRrnGl8MOjkbLsWNWRHYJJN9yYBdXkaLKKTvNpjLiC)
    - **DataSet** : [`Book Crossing`](https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset) [`MovieLens`](https://grouplens.org/datasets/movielens/) [`Netflix Prize`](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) [`Yahoo! Music`](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r)

- **FISM** [`REVIEW`]()
    - **Paper** : ["FISM: Factored Item Similarity Models for Top-N Recommender Systems", Kabbur et al., KDD, 2013](https://dl.acm.org/doi/abs/10.1145/2487575.2487589?casa_token=zDZvzz_byroAAAAA:1Dr1GXJ7yst1AM9GKAlEyDRP6_hzDEQQr5ML9cjR7u6bJOr4dOp4gA3RyLyI-tVdsewY6FL7Sixq4Vs)
    - **DataSet** : [`MovieLens`](https://grouplens.org/datasets/movielens/) [`Netflix Prize`](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) [`Yahoo! Music`](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r)

</br>

### Latent Factor Model leveraging Deep Learning Techs

#### `2024.04.03.` Neural Collaborative Filtering

- **NCF** [`REVIEW`]() [`CODE`](https://github.com/jayarnim/MD-Data_Mining_Lab/blob/main/model/NCF.py)

    - **Paper** : ["Neural Collaborative Filtering", He et al., WWW, 2017](https://dl.acm.org/doi/abs/10.1145/3038912.3052569?casa_token=xJcQ62dMU8kAAAAA:erA0iE1l2Pxdx8qpbMFCh7Z6-qc02h-yCXcoaWJN5E4pJwMwu6RVRoMrBdUSFJ_yrHGdTfVtJR67EPw)
    - **DataSet** : [`MovieLens`](https://grouplens.org/datasets/movielens/)

#### `2024.04.17.` AutoEncoder based Latent Factor Model

- **AutoRec** [`REVIEW`]()
    - **Paper** : ["AutoRec: Autoencoders Meet Collaborative Filtering", Sedhain et al., WWW, 2015](https://dl.acm.org/doi/abs/10.1145/2740908.2742726?casa_token=h6-W8fBHMuwAAAAA:hcZXeeqXUng_hrZJ9GaPt3dfJ4lXKK_THtypbucIf-XV18hRNfMxj2CkZKTOShkdwCCcrJ5WEGho-mo)
    - **DataSet** : [`MovieLens`](https://grouplens.org/datasets/movielens/)

- **Deep AutoRec** [`REVIEW`]() [`CODE`](https://github.com/jayarnim/M-Data_Mining_Lab/blob/main/model/DeepAutoRec.py)
    - **Paper** : [“Training Deep AutoEncoders for Collaborative Filtering”, Kuchaiev and Ginsburg, 2017](https://arxiv.org/abs/1708.01715)
    - **DataSet** : [`Netflix Prize`](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

</br>

### Hybrid Latent Factor Model

#### `2024.05.01.` Factorization Machine Series

- **FM** [`REVIEW`]()
    - **Paper** : ["Factorization Machines", Rendle, ICDM, 2010](https://ieeexplore.ieee.org/abstract/document/5694074?casa_token=PxTxcXYbSBEAAAAA:94LVL0iDWaWBXagioWFO-JagI4rp2mGkpcl-agJtPsKwhs7WhMS-f5mitp-OrI5z8M2bcAUrzLBR)
    - **DataSet** : [`Netflix Prize`](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) [`ECML/PKDD Discovery Challenge 2009`](https://www.kde.cs.uni-kassel.de/wp-content/uploads/ws/dc09/dataset.html)

- **FFM** [`REVIEW`]()
    - **Paper** : ["Field-aware Factorization Machines for CTR Prediction", Juan et al., RecSys, 2016](https://dl.acm.org/doi/abs/10.1145/2959100.2959134?casa_token=LhyqvBbTAH4AAAAA:j1IOKYkeCTiByjmyaTueiRLCZkmi5U0SWqEVOyBbOdZOj9xKlu7X8AeBWPsum8IwcP6hUdTHqvJgfcM)
    - **DataSet** : [`Display Advertising Challenge`](https://www.kaggle.com/c/criteo-display-ad-challenge/data) [`Click-Through Rate Prediction`](https://www.kaggle.com/c/avazu-ctr-prediction/data)

- **NFM** [`REVIEW`]() [`CODE`](https://github.com/jayarnim/M-Data_Mining_Lab/blob/main/model/NeuralFactorizationMachine.py)
    - **Paper** : ["Neural Factorization Machines for Sparse Predictive Analytics", He and Chua, SIGIR, 2017](https://dl.acm.org/doi/abs/10.1145/3077136.3080777?casa_token=GwAdLrQPwy4AAAAA:ie1lvyHs54HbZmQS4pns-P585Knu3QIYRcNXUbPbfyQdNIO-E2HGXQCIwoza5np_wt-S4gs1lcQ_yw4)
    - **DataSet** : [`Frappe`](https://www.baltrunas.info/context-aware) [`MovieLens`](https://grouplens.org/datasets/movielens/)

- **DeepFM** [`REVIEW`]()
    - **Paper** : ["DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", Guo et al., IJCAI, 2017](https://arxiv.org/abs/1703.04247)
    - **DataSet** : [`Display Advertising Challenge`](https://www.kaggle.com/c/criteo-display-ad-challenge/data)

- **AFM** [`REVIEW`]()
    - **Paper** : ["Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks", Xiao et al., IJCAI, 2017](https://arxiv.org/abs/1708.04617)
    - **DataSet** : [`Frappe`](https://www.baltrunas.info/context-aware) [`MovieLens`](https://grouplens.org/datasets/movielens/)

</br>

### Temporal RecSys

#### `2024.05.22.` Temporal Latent Factor Model

- **TimeSVD++** [`REVIEW`]()

    - **Paper** : ["Collaborative Filtering with Temporal Dynamic", Koren, KDD, 2009](https://dl.acm.org/doi/abs/10.1145/1557019.1557072?casa_token=rimTEX65IIsAAAAA:jfa7Vyrl6bt4D3OsxxBP2ja1FfR6DbK7EjnoTTgxbddG16QaJzh0QTSTppGwWkaJVG0nvRMba_jmB3w)
    - **DataSet** : [`Netflix Prize`](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

</br>

### Sequencial RecSys

#### `2024.06.05.` Markov Chains based Sequencial Latent Factor Model

- **FPMC** [`REVIEW`]()

    - **Paper** : ["Factorizing personalized Markov chains for next-basket recommendation", Rendle et al., WWW, 2010](https://dl.acm.org/doi/abs/10.1145/1772690.1772773?casa_token=Q3sHZL_spjgAAAAA:2Xm7ovGfhXZSkNb2ulgWO27DY0vMDKkoVrQS23pMKqouoJS1y_AKVeQlCMI_tCsuyggAGMY-IgYrXeU)
    - **DataSet** : [`Rossman Store Sales`](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales)

#### `2024.06.19.` RNN based Sequencial RecSys

- **GRU4REC** [`REVIEW`]() [`CODE`]()

    - **Paper** : ["Session-based Recommendations with Recurrent Neural Networks", Hidasi et al., ICLR, 2016](https://arxiv.org/abs/1511.06939)
    - **DataSet** : [`RecSys Challenge 2015`](https://www.kaggle.com/code/danofer/2015-recsys-challenge-starter)

#### `2024.06.26.` Self Attention Mechanism based RecSys

- **SASREC** [`REVIEW`]() [`CODE`]()

    - **Paper** : ["Self-Attentive Sequential Recommendation", Kang and McAuley, ICDM, 2018](https://ieeexplore.ieee.org/abstract/document/8594844?casa_token=JT5smtt5Z5sAAAAA:lFfXP_q_01zzLRSEc7p1zEyR_jZ7l1VjeTTCOUO6QMkDmw6HUM0BDtBSnPGpvH6XZmxvQwnGi-r7)
    - **DataSet** : [`Steam Store Sales`](https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset) [`Amazon Beauty`](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) [`Amazon VideoGame`](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) [`MovieLens`](https://grouplens.org/datasets/movielens/)

#### `2024.07.03.` BERT based Sequencial RecSys

- **BERT4REC** [`REVIEW`]() [`CODE`]()

    - **Paper** : ["BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer", Sun et al., CIKM, 2019](https://dl.acm.org/doi/abs/10.1145/3357384.3357895?casa_token=FdOnUIipxhwAAAAA:jXWonRcvhqi5WJFCb_hKPdJMAWgvZI9YJzI4qn20pSMM7N6FrxdvcL9g9h1pAibEFy5eiD_z4N9XmbE)
    - **DataSet** : [`Steam Store Sales`](https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset) [`Amazon Beauty`](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) [`MovieLens`](https://grouplens.org/datasets/movielens/)

</br>

### RecSys leveraging Generative Model

#### `2024.07.10.` VAE based Latent Factor Model

- **Mult-VAE** [`REVIEW`]() [`CODE`]()

    - **Paper** : ["Variational Autoencoders for Collaborative Filtering", Liang et al., WWW, 2018](https://dl.acm.org/doi/abs/10.1145/3178876.3186150)
    - **DataSet** : [`MovieLens`](https://grouplens.org/datasets/movielens/) [`Netflix Prize`](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) [`Million Song Dataset`](https://www.kaggle.com/datasets/ryanholbrook/the-million-songs-dataset)

#### `2024.07.17.` GAN based Collaborative Filtering

- **IRGAN** [`REVIEW`]() [`CODE`]()

    - **Paper** : ["IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models", Wang et al., SIGIR, 2017](https://dl.acm.org/doi/abs/10.1145/3077136.3080786?casa_token=l3DUV8WZZPUAAAAA:gh1OnSEylDd-KiNnTyq2jTgCcIAutcHOYKgFk9rXXmzdy8t8lJjfYi0XJDVzEVIsENZs8wlTCZeN_Wc)
    - **DataSet** : [`LETOR 4.0`](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) [`MovieLens`](https://grouplens.org/datasets/movielens/) [`Netflix Prize`](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) [`InsuranceQA`](https://github.com/shuzi/insuranceQA)

</br>

### Graph based RecSys

#### `2024.07.24.` PageRank based RecSys

- **Pixie** [`REVIEW`]() [`CODE`]()

    - **Paper** : ["Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time", Eksombatchai et al., WWW, 2018](https://dl.acm.org/doi/abs/10.1145/3178876.3186183)

#### `2024.07.31.` DeepWalk based RecSys

- **EGES** [`REVIEW`]() [`CODE`]()

    - **Paper** : [“Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba”, Wang et al., KDD, 2018](https://dl.acm.org/doi/abs/10.1145/3219819.3219869?casa_token=GZdt3pMrclMAAAAA:3ibqOAtkUJToggQLyiDIeqhX9HqhZrvkkGoH8NX2bEEPJLsaydyR6qgVLEiaaut5S3zHImnY189XDx4)
    - **DataSet** : [`Amazon Electronics`](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)

#### `2024.08.07.` GraphSAGE based RecSys

- **PinSAGE** [`REVIEW`]() [`CODE`]()
    - **Paper** : ["Graph Convolutional Neural Networks for Web-Scale Recommender Systems", Ying et al., KDD, 2018](https://dl.acm.org/doi/abs/10.1145/3219819.3219890?casa_token=Au-umXQUZ1kAAAAA:lJzYsga18v6bN9pxyApAxnegROTbuvoCB8ukqZ3A8NiPKxY7sfXdSHsvu4eCIWgtQFoS0AaZFSzjHHY)
    - **DataSet** : 

- **NGCF** [`REVIEW`]() [`CODE`]()
    - **Paper** : ["Neural Graph Collaborative Filtering", Wang et al., SIGIR, 2019](https://dl.acm.org/doi/abs/10.1145/3331184.3331267?casa_token=8JTOV4RxYlsAAAAA:bkwRnHjoNWGcx5bGw97-cRpFT4iKhBSLnEyI3xK0eXEsb2-bLIwANoE1txFvyRCsgpABkhbCzrtjRA)
    - **DataSet** : 

#### `2024.08.14.` Knowledge Graph based RecSys

- **RippleNet** [`REVIEW`]() [`CODE`]()
    - **Paper** : [“RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems”, Wang et al., CIKM, 2018](https://dl.acm.org/doi/abs/10.1145/3269206.3271739?casa_token=R1-vKJgCzrsAAAAA:x-U83HRTCb83izvU4lkdL29VKSeUgBBgFOpgWmjwpsa6PGdjVig-jaoUI6YdzKY6LihmfGshjhcp2Ks)
    - **DataSet** : 

- **KGAT** [`REVIEW`]() [`CODE`]()
    - **Paper** : ["KGAT: Knowledge Graph Attention Network for Recommendation", Wang et al., KDD, 2019](https://dl.acm.org/doi/abs/10.1145/3292500.3330989?casa_token=H-IaOAQVwHwAAAAA:2299fELWgPC7Y7f14vmWKDt0ZhrWV3I01NYuM6s1CoOyEwrltgYDzs1jP6GK_zU6v5qiwXHByDAqmIQ)
    - **DataSet** : 

</br>

## Reference

- 추천 시스템, 차루 아가르왈 저

- [딥러닝을 활용한 추천 시스템 구현](https://fastcampus.co.kr/data_online_rs), [이재원](https://github.com/jaewonlee-728), 패스트캠퍼스

- [30개 프로젝트로 끝내는 추천 시스템 초격차 패키지](https://fastcampus.co.kr/data_online_rsystem), 패스트캠퍼스
