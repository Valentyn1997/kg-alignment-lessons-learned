# kg-alignment-lessons-learned
Implementation of reproducibility paper "Knowledge Graph Entity Alignment with Graph Convolutional Networks: Lessons Learned"

Link to arXiv: TODO

## Instalation
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
