# Learning to Perturb Word Embeddings for Out-of-distribution QA
This is the Pytorch implementation for the paper **Learning to Perturb Word Embeddings for Out-of-distribution QA** (**ACL 2021**): [[Paper]](https://arxiv.org/abs/2105.02692)

## Abstract
<img align="middle" width="900" src="https://github.com/seanie12/SWEP/blob/main/images/concept_fig.png">

QA models based on pretrained language mod-els have achieved remarkable performance onv arious benchmark datasets.However, QA models do not generalize well to unseen data that falls outside the training distribution, due to distributional shifts.Data augmentation (DA) techniques which drop/replace words have shown to be effective in regularizing the model from overfitting to the training data.Yet, they may adversely affect the QA tasks since they incur semantic changes that may lead to wrong answers for the QA task. To tackle this problem, we propose a simple yet effective DA method based on a stochastic noise generator, which learns to perturb the word embedding of the input questions and context without changing their semantics. We validate the performance of the QA models trained with our word embedding perturbation on a single source dataset, on five different target domains.The results show that our method significantly outperforms the baselineDA methods. Notably, the model trained with ours outperforms the model trained with more than 240K artificially generated QA pairs.

__Contribution of this work__
- We propose a simple yet effective data augmentation method to improve the generalization performance of pretrained language models for QA tasks.
- We show that our learned input-dependent perturbation function transforms the original input without changing its semantics, which is
crucial to the success of DA for question answering.
- We extensively validate our method for domain generalization tasks on diverse datasets,
on which it largely outperforms strong baselines, including a QA-pair generation method.



# Reference
To cite the code/paper, please use this BibTex
```bibtex
@inproceedings{lee2021learning,
  title={Learning to Perturb Word Embeddings for Out-of-distribution QA},
  author={Lee, Seanie and Kang, Minki and Lee, Juho and Hwang, Sung Ju},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```


## Dependencies
This code is written in Python. Dependencies include
* python == 3.6
* pytorch == 1.4
* json-lines
* tqdm
* transformes == 3.0.2


## How to train the model
```bash
python run_squad.py --read_data --train_file "squad-train file" --dev_file "dev-squad file" --model_dir "directory for model checkpoint"
```

## Download data for SQuAD
```bash
mkdir squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O ./squad/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./squad/dev-v1.1.json
```

## Download pickle file for training SQuAD
We provide preprocessed file of SQuAD dataset. Download tar.gz file from [here](https://drive.google.com/file/d/1cn52AG6mVviWt-zRXO8QyXXyex3rbCUi/view?usp=sharing) and unzip it at the root directory.

## Download BioASQ
```bash
mkdir bio-asq
wget http://participants-area.bioasq.org/MRQA2019/ -O ./bio-asq/BioASQ.jsonl.gz
```


## Download the other datasets
```bash
mkdir shift-data
```
You can down load the dataset from [here](https://modestyachts.github.io/squadshifts-website/)
and put it under the directory "shift-data".

## Evaluation of bio-asq
```bash
python eval_bio.py --ckpt_file "file path for model checkpoint" --output_dir "directory for evaluation result"
```

## Evaluation of the other dataset

```bash
python eval_shift.py --ckpt_file "model checkpoint" --output_dir "directory for evaluation result"
```
