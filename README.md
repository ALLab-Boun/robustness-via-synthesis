# Robustness-via-Synthesis: Robust training with generative adversarial perturbations
## Introduction

This [paper](https://www.sciencedirect.com/science/article/pii/S0925231222013091?via%3Dihub) presents a robust training algorithm where the adversarial perturbations are automatically synthesized from a random vector using a generator network. The classifier is trained with cross-entropy loss regularized with the optimal transport distance between the representations of the natural and synthesized adversarial samples. The proposed approach attains comparable robustness with various gradient-based and generative robust training techniques on CIFAR10, CIFAR100, SVHN, and Tiny ImageNet datasets. Code for CIFAR10 is provided in this repository. The codebase is modified from [MadryLab's cifar10_challenge](https://github.com/MadryLab/cifar10_challenge.git). Pretrained models for CIFAR10 and CIFAR100 are also shared.

# Usage
For training
```
python train.py
```
For evaluation:
```
python eval_pgd_attack.py
```
# Data and Models
cifar10_data folder and trained models for CIFAR10 and CIFAR100 can be accessed through the following link.

https://drive.google.com/drive/folders/1F6hHz1WbymE6w2hzWf6nNqMbd9-UvS5R?usp=share_link

# Cite
If you find this work is useful, please cite the following:
```
@article{baytacs2022robustness,
  title={Robustness-via-synthesis: Robust training with generative adversarial perturbations},
  author={Bayta{\c{s}}, {\.I}nci M and Deb, Debayan},
  journal={Neurocomputing},
  year={2022},
  publisher={Elsevier}
}
```
