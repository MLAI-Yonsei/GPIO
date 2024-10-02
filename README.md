#  Graph Perceiver IO: A General Architecture for Graph-Structured Data

### Abstract
Multimodal machine learning has been widely studied for the development of general intelligence. Recently, the Perceiver and Perceiver IO, show competitive results for diverse dataset domains and tasks. However, recent works, Perceiver and Perceiver IO, have focused on heterogeneous modalities, including image, text, and there are few research works for graph structured datasets. A graph has an adjacency matrix different from other datasets such as text and image, and it is not trivial to handle the topological information, relational information, and canonical positional information. In this study, we provide a Graph Perceiver IO (GPIO), the Perceiver IO for the graph structured dataset. We keep the main structure of the GPIO as the Perceiver IO because the Perceiver IO already handles the diverse dataset well, except for the graph structured dataset. The GPIO is a general method, and it handle diverse datasets such as graph structured data as well as text and images. Compared to the graph neural networks, the GPIO requires a lower complexity, and it can incorporate the global and local information efficiently. Furthermore, we propose GPIO+ for the multimodal few-shot classification that incorporates both images and graphs simultaneously.

### Authors
Seyun Bae, Hoyoon Byun, Changdae Oh, Yoon-Sik Cho, Kyungwoo Song

### Run
For each task, you can create and run a bash file like the code below. Below is an example of node classification. 

```
python main.py \
--l_d=32 \
--n_l=16 \
--l_dh=128 \
--c_dh=128 \
--l_h=16 \
--depth=1 \
--lr=5e-4 \
--wd=1e-2 \
--c_h=32 \
--p_dim=0 \
--random_splits=True \
--epoch=200 \
--dataset=Cora \
--seed=2025 ;
`````

We referred to the code provided below.

* **[RWPE](https://github.com/vijaydwivedi75/gnn-lspe)**
* **[VGNAE](https://github.com/SeongJinAhn/VGNAE)**
* **[PyG](https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark)**
* **[Perceiver IO](https://github.com/lucidrains/perceiver-pytorch)**
* **[EGNN](https://github.com/jmkim0309/fewshot-egnn)**

Datasets used in graph classification, node classification and link prediction can be automatically downloaded from pytorch geometric.

Datasets used in few-shot classification can be downloaded from public source code of EGNN.

Hyperparameter settings for each task and dataset is described in supplementary material.

