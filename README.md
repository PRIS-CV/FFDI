# FFDI
## Introduction
Code release for "Domain Generalization via Frequency-domain-based Feature Disentanglement and Interaction" (ACM MM 2022): [https://arxiv.org/abs/2201.08029](https://arxiv.org/abs/2201.08029)

Part of the code is inherited from [Episodic-DG](https://github.com/HAHA-DL/Episodic-DG).

## 1. Requirements:
'''
GPU GeForce RTX 1080 Ti
pytorch==1.9.0
torchvision==0.10.0
cudatoolkit==10.2.89
opencv-python==4.5.2.54
h5py==2.10.0
scikit-learn==0.24.2
pillow==7.1.2
scipy==1.7.0
numpy==1.19.2
'''

## 2. Prepare:
### Datasets:
Please download the [PACS](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ&usp=sharing) datasets and use the official train/val split.

### ImageNet pretrained model
We use the pytorch pretrained ResNet-18 model from [https://download.pytorch.org/models/resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth)

## 3. Training:
- Train from scratch with command:
```bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_agg.py

```

## Citing the related works

Please cite the related works in your publications if it helps your research:

    @article{wang2022domain,
      title={Domain Generalization via Frequency-based Feature Disentanglement and Interaction},
      author={Wang, Jingye and Du, Ruoyi and Chang, Dongliang and Liang, KongMing and Ma, Zhanyu},
      journal={arXiv preprint arXiv:2201.08029},
      year={2022}
    }

    @InProceedings{Li_2019_ICCV,
      author = {Li, Da and Zhang, Jianshu and Yang, Yongxin and Liu, Cong and Song, Yi-Zhe and Hospedales, Timothy M.,
      title = {Episodic Training for Domain Generalization},
      booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
      month = {October},
      year = {2019}
    }


    

