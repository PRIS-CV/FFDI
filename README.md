# FFDI
## Introduction
Code release for "Domain Generalization via Frequency-domain-based Feature Disentanglement and Interaction" (ACM MM 2022): [https://arxiv.org/abs/2201.08029](https://arxiv.org/abs/2201.08029).

Part of the code is inherited from [Episodic-DG](https://github.com/HAHA-DL/Episodic-DG).

## Enviroments
```bash
GPU GeForce RTX 1080 Ti
pytorch==1.9.0
torchvision==0.10.0
cudatoolkit==10.2.89
```

## Prepare
### Datasets
Please download the [PACS](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ&usp=sharing) datasets and use the official train/val split.

### ImageNet pretrained model
We use the pytorch pretrained ResNet-18 model from [https://download.pytorch.org/models/resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth).

## Run
- Train from scratch with command:
```bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_agg.py

```

## Reference

Please cite the related works in your publications if it helps your research:

    @inproceedings{wang2022domain,
      title={Domain Generalization via Frequency-domain-based Feature Disentanglement and Interaction},
      author={Wang, Jingye and Du, Ruoyi and Chang, Dongliang and Liang, KongMing and Ma, Zhanyu},
      booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
      year={2022}
    }


    

