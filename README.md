# ECBM E4040 Final Project - A Reimplementation of the 'Swin' Transformer using Tensorflow 2.x

This repo contains the code for a Tensorflow 2.x implementation of the Swin Transformer from the paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030). Additionally, the transformer model is evaluated on the CIFAR-10 Image Classification Dataset.

![Swin Transformer Architecture proposed by the original paper authors](https://www.section.io/engineering-education/an-overview-of-swin-transformer/swin-transformer.png)

## Model weights

## Organization of this directory

```
./
├── Efficientnet_Comparison.ipynb
├── README.md
├── Resnet_Comparison.ipynb
├── Swin Transformer Classification on CIFAR-10.ipynb
├── ViT_Comparison.ipynb
├── model-checkpoints (Excluded from github repo)
│   ├── Efficientnet
│   │   ├── EfficientNetB3-cifar10.data-00000-of-00001
│   │   ├── EfficientNetB3-cifar10.index
│   │   └── checkpoint
│   ├── Resnet
│   │   ├── Resnet50-cifar10.data-00000-of-00001
│   │   ├── Resnet50-cifar10.index
│   │   └── checkpoint
│   ├── Swin
│   │   ├── Swin-cifar10.data-00000-of-00001
│   │   ├── Swin-cifar10.index
│   │   └── checkpoint
│   └── ViT
│       ├── ViT-cifar10.data-00000-of-00001
│       ├── ViT-cifar10.index
│       └── checkpoint
├── runs
│   ├── EfficientnetB4-cifar10
│   │   ├── train
│   │   │   └── events.out.tfevents.1671408748.84367d077987.2125.0.v2
│   │   └── validation
│   │       └── events.out.tfevents.1671408844.84367d077987.2125.1.v2
│   ├── Resnet50-cifar10
│   │   ├── train
│   │   │   └── events.out.tfevents.1671403885.84367d077987.129.11.v2
│   │   └── validation
│   │       └── events.out.tfevents.1671403969.84367d077987.129.12.v2
│   ├── Swin-cifar10
│   │   ├── train
│   │   │   └── events.out.tfevents.1671398407.nndl-a-bkup-1.10913.0.v2
│   │   └── validation
│   │       └── events.out.tfevents.1671398804.nndl-a-bkup-1.10913.1.v2
│   └── ViT-cifar10
│       ├── train
│       │   └── events.out.tfevents.1671409577.76e0414d7507.505.8.v2
│       └── validation
│           └── events.out.tfevents.1671409703.76e0414d7507.505.9.v2
└── utils
    ├── model_vit.py 
    ├── callback.py
    ├── layer_funcs.py
    ├── layers.py
    └── model.py

20 directories, 33 files
```
