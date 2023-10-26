# EMSCNet
EMSCNet: Efficient Multi-Sample Contrastive Network for Remote Sensing Image Scene Classification

Here are some tips for running EMSCNet:  
1. AID, NWPU, and UCM data sets should be placed in the data directory.  
2. We provide the training code for the AID dataset under EMSCNet (ViT-B) in train.py. Please note that the pre-trained weights on ImageNet should be downloaded from the link below and placed into the home directory before running.  
   ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).  
   ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.  
   baidu_url: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA password: eu9f  
4. If you have other problems about the article or the codes, please contact us by this email: zhaoyibo2027@163.com. Thanks for your support.

**Bibtex**
```
@ARTICLE{10086539,
  author={Zhao, Yibo and Liu, Jianjun and Yang, Jinlong and Wu, Zebin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={EMSCNet: Efficient Multisample Contrastive Network for Remote Sensing Image Scene Classification}, 
  year={2023},
  volume={61},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2023.3262840}}
```
