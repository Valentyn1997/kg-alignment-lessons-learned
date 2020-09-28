# [Knowledge Graph Entity Alignment with Graph Convolutional Networks: Lessons Learned](https://doi.org/10.1007/978-3-030-45442-5_1)

<a href="https://doi.org/10.1007/978-3-030-45442-5_1"><img src="https://img.shields.io/badge/DOI-10.1007/978--3--030--45442--5__1-fcb426" alt=""></a> <a href="https://arxiv.org/abs/1911.08342"><img src="https://img.shields.io/badge/arXiv-1911.08342-b31b1b"/></a>


## Explore results
You can view the results of all experiments in [Experiments.ipynb](https://github.com/Valentyn1997/kg-alignment-lessons-learned/blob/master/notebooks/Experiments.ipynb).

## Installation
1. Make sure, you have Python 3.7
2. ```sudo pip3 install torch```
3. ```sudo pip3 install -r requirements.txt```


## Run
If you want to use [mlflow](https://mlflow.org/), start mlflow server first:  

```mlflow server```

Then, you can run either simple model training or hyperparameter search:

```bash
cd src/
PYTHONPATH=. python3  gcn_align_runnable.py --log_to_mlflow  --model=GCNAlign --dataset_name=dbp15k_jape --subset_name=zh_en
```
## Citing

If you find this repository useful please consider citing our paper
```bibtex
@InProceedings{10.1007/978-3-030-45442-5_1,
author="Berrendorf, Max and Faerman, Evgeniy and Melnychuk, Valentyn and Tresp, Volker and Seidl, Thomas",
editor="Jose, Joemon M. and Yilmaz, Emine and Magalh{\~a}es, Jo{\~a}o and Castells, Pablo and Ferro, Nicola and Silva, M{\'a}rio J. and Martins, Fl{\'a}vio",
title="Knowledge Graph Entity Alignment with Graph Convolutional Networks: Lessons Learned",
booktitle="Advances in Information Retrieval",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="3--11",
abstract="In this work, we focus on the problem of entity alignment in Knowledge Graphs (KG) and we report on our experiences when applying a Graph Convolutional Network (GCN) based model for this task. Variants of GCN are used in multiple state-of-the-art approaches and therefore it is important to understand the specifics and limitations of GCN-based models. Despite serious efforts, we were not able to fully reproduce the results from the original paper and after a thorough audit of the code provided by authors, we concluded, that their implementation is different from the architecture described in the paper. In addition, several tricks are required to make the model work and some of them are not very intuitive.We provide an extensive ablation study to quantify the effects these tricks and changes of architecture have on final performance. Furthermore, we examine current evaluation approaches and systematize available benchmark datasets.We believe that people interested in KG matching might profit from our work, as well as novices entering the field. (Code: https://github.com/Valentyn1997/kg-alignment-lessons-learned).",
isbn="978-3-030-45442-5"
}
```
