# End-to-end weakly-supervised semantic alignment

![](http://www.di.ens.fr/willow/research/weakalign/images/teaser.jpg)


## About

This is the implementation of the paper "End-to-end weakly-supervised semantic alignment" by I. Rocco, R. Arandjelović and J. Sivic. 

For more information check out the project [[website](http://www.di.ens.fr/willow/research/weakalign/)] and the paper on [[arXiv](https://arxiv.org/abs/1712.06861)].


## Getting started

### Dependencies

The code is implemented using Python 3 and PyTorch 0.2. All dependencies are included in the standard Anaconda distribution.

### Training

The code includes scripts for pre-training the models with strong supervision (`train_strong.py`) as proposed in [our previous work](http://www.di.ens.fr/willow/research/cnngeometric/), as well as to fine-tune the model using weak supervision (`train_weak.py`) as proposed in this work.

Training scripts can be found in the `scripts/` folder.

### Evaluation

Evaluation is implemented in the `eval.py` file. It can evaluate a single affine or TPS model (with the `--model-aff` and `--model-tps` parameters respectively), or a combined affine+TPS model (with the `--model`) parameter.

The evaluation dataset is passed with the `--eval-dataset` parameter.

### Trained models

Trained models for the baseline method using only strong supervision and the proposed method using additional weak supervision are provided below. You can store them in the `trained_models/` folder. 

With the provided code below you should obtain the results from Table 2 of the paper.


**CNNGeometric with VGG-16 baseline:** [[affine model](http://www.di.ens.fr/willow/research/weakalign/trained_models/cnngeo_vgg16_affine.pth.tar)],[[TPS model](http://www.di.ens.fr/willow/research/weakalign/trained_models/cnngeo_vgg16_tps.pth.tar)]

```
python eval.py --feature-extraction-cnn vgg --model-aff trained_models/cnngeo_vgg16_affine.pth.tar --model-tps trained_models/cnngeo_vgg16_tps.pth.tar --eval-dataset pf-pascal
```

**CNNGeometric with ResNet-101 baseline:** [[affine model](http://www.di.ens.fr/willow/research/weakalign/trained_models/cnngeo_resnet101_affine.pth.tar)],[[TPS model](http://www.di.ens.fr/willow/research/weakalign/trained_models/cnngeo_resnet101_tps.pth.tar)]

```
python eval.py --feature-extraction-cnn resnet101 --model-aff trained_models/cnngeo_resnet101_affine.pth.tar --model-tps trained_models/cnngeo_resnet101_tps.pth.tar --eval-dataset pf-pascal
```

**Proposed method:** [[combined aff+TPS model](http://www.di.ens.fr/willow/research/weakalign/trained_models/weakalign_resnet101_affine_tps.pth.tar)]

```
python eval.py --feature-extraction-cnn resnet101 --model trained_models/weakalign_resnet101_affine_tps.pth.tar --eval-dataset pf-pascal
```

## BibTeX 

If you use this code in your project, please cite our paper:
````
@article{Rocco18,
        author       = "Rocco, I. and Arandjelovi\'c, R. and Sivic, J.",
        title        = "End-to-end weakly-supervised semantic alignment",
        journal={arXiv preprint arXiv:1712.06861},
         }
````


