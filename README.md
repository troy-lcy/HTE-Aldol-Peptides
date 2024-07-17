# Peptide-Aldol-Reaction

<div align="center">
    <img src='https://github.com/troy-lcy/HTE-Aldol-Peptides/blob/main/Figures/TOC.png' style="width:400px">
</div>

## Highlights

In this work, we propose a novel framework that utilizes a pre-trained molecular properties model (Uni-Mol) for rapid generation of molecular feature representations and integrates machine learning models for reaction prediction. 
1. We tested this framework on three different high-throughput experimentation (HTE) reaction datasets, and demonstrate results comparable to expert-designed DFT descriptors.
2. We synthesized a series of tetrapeptides that catalyze asymmetric aldol reactions under HTE conditions to generate a small-size dataset. 
3. With Uni-Mol representations, We trained a random forest classifier model to identify tetrapeptide catalysts exhibiting high yield and enantioselectivity.

## Comparing with other molecular descriptor methods

<div align="center">
    <img src='https://github.com/troy-lcy/HTE-Aldol-Peptides/blob/main/Figures/Fig1.png' style="width:600px">
</div>

## Installation
Python>3.8 is recommanded.
To install `Uni-Mol` related tools, please refer to:
[Uni-Mol Link](https://github.com/deepmodeling/Uni-Mol)

@inproceedings{
  zhou2023unimol,
  title={Uni-Mol: A Universal 3D Molecular Representation Learning Framework},
  author={Gengmo Zhou and Zhifeng Gao and Qiankun Ding and Hang Zheng and Hongteng Xu and Zhewei Wei and Linfeng Zhang and Guolin Ke},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=6K2RM6wVqKu}
}

Other required packages includes: 
[Sklearn](https://github.com/scikit-learn/scikit-learn), [Matplotlib](https://github.com/matplotlib/matplotlib), [UMAP](https://umap-learn.readthedocs.io/en/latest/)
