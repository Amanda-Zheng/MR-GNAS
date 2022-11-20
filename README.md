# MR-GNAS: Multi-Relational Graph NAS with Fine-grained MP

The official Pytorch implementation of ICDM'22 "Multi-Relational Graph Neural Architecture Search with Fine-grained Message Passing"

To tackle the limitations of single-relational setting and coarse-grained search space design in existing graph NAS, in this paper, we propose a novel framework of multirelational graph neural architecture search, dubbed MR-GNAS, to automatically develop innovative and excellent multi-relational GNN architectures.

## Framework
![image](https://github.com/Amanda-Zheng/MR-GNAS/blob/9bdb73b26104cbea78adf410ca0e234ca88e1fb7/mr-gnas.png)


## Instructions
Here provide some examples of how to run the code (with link prediction task, same with node classification task), and the best hyper-parameters can be found in our work.
### step 1: search in the proposed search space

```python 
python mr_lp_search.py
```

### step 2: train with searched architectures
```python 
python mr_lp_train.py --genotype "[Genotype(alpha_cell=[('pre_sub', 1, 0), ('f_sparse_comp', 2, 1), ('f_sparse_comp', 3, 2), ('a_max', 4, 2), ('a_max', 5, 3), ('f_sparse_last', 6, 5), ('f_sparse_last', 7, 5)], concat_node=[4, 5, 6, 7], score_func='sf_DisMult')]"
```
## Discussion & Cite
Please feel free to connect xin.zheng@monash.edu for any questions and issues of this work.

You are welcome to kindly cite our paper:

```
@inproceedings{mrgnas_zheng2022mrgnas,
  title={Multi-Relational Graph Neural Architecture Search with Fine-grained Message Passing},
  author={Zheng, Xin and Zhang, Miao and Chen, Chunyang and Li, Chaojie and Zhou, Chuan, and Pan, Shirui},
  booktitle={ICDM},
  year={2022}
}
```
