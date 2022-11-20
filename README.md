# [ICDM-22] MR-GNAS

Official Pytorch implementation of ICDM'22 "Multi-Relational Graph Neural Architecture Search with Fine-grained Message Passing"

To tackle the limitations of single-relational setting and coarse-grained search space design in existing graph NAS, in this paper, we propose a novel framework of multirelational graph neural architecture search, dubbed MR-GNAS, to automatically develop innovative and excellent multi-relational GNN architectures.

## Framework<img src="/Users/xinzheng/Downloads/mr-gnas.png"/>


## Instructions
Here provide some examples of how to run the code (with link prediction task, same with node classification task), and the best hyper-parameters can be found in our work.
### step 1: search in the proposed search space

`python mr_lp_search.py --dataset FB15k-237 --device cuda:0 \
--layers 2 --init_fea_dim 128 --feature_dim 128 --seed 202206 \
--first_nodes 1 --last_nodes 1 \
--epochs 10000 --arch_learning_rate 5e-4 --learning_rate 1e-3  --learning_rate_min 1e-4 \
--graph_batch_size 30000 --graph_batch_size_val 10000 --max_patience 500`

### step 2: train with searched architectures
`python mr_lp_train.py --data FB15k-237 --device cuda:0 \
--batch_size 512 --learning_rate 0.001 --learning_rate_min 5e-4 --epoch 1500 --num_base_r 475 --init_fea_dim 200 --feature_dim 200 --save_model_freq 299 \
--drop_op 0.4 \
--genotype \
"[Genotype(alpha_cell=[('pre_sub', 1, 0), ('f_sparse_comp', 2, 1), ('f_sparse_comp', 3, 2), ('a_max', 4, 2), ('a_max', 5, 3), ('f_sparse_last', 6, 5), ('f_sparse_last', 7, 5)], concat_node=[4, 5, 6, 7], score_func='sf_DisMult')]"
`
## Discussion & Cite
Please feel free to connect xin.zheng@monash.edu for any questions and issues of this work.

You are welcome to kindly cite our paper:

`@inproceedings{mrgnas_zheng2022mrgnas,
  title={Multi-Relational Graph Neural Architecture Search with Fine-grained Message Passing},
  author={Zheng, Xin and Zhang, Miao and Chen, Chunyang and Li, Chaojie and Zhou, Chuan, and Pan, Shirui},
  booktitle={ICDM},
  year={2022}
}`