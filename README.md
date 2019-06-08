# [Adversarial Networks for Cross-Modal Food Retrieval](https://arxiv.org/abs/1905.01273)
## Codes of ACME (PyTorch)
*Learning Cross-Modal Embeddings with Adversarial Networks for Cooking Recipes and Food Images*  
Wang Hao, Doyen Sahoo, Chenghao Liu, Ee-peng Lim, Steven C. H. Hoi   
CVPR 2019  

![outline](https://github.com/hwang1996/ACME/blob/master/imgs/cvpr_fig.png)

If you find this code useful, please consider citing:
```
@inproceedings{wang2019learning,
  title={Learning Cross-Modal Embeddings With Adversarial Networks for Cooking Recipes and Food Images},
  author={Wang, Hao and Sahoo, Doyen and Liu, Chenghao and Lim, Ee-peng and Hoi, Steven CH},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11572--11581},
  year={2019}
}
```
Our work is an extension of [im2recipe](https://github.com/torralba-lab/im2recipe-Pytorch), where you can borrow some food data pre-processing methods.

## Installation
We use pytorch v0.5.0 and python 3.5.2 in our experiments.  
You need to download the Recipe1M dataset from [here](http://im2recipe.csail.mit.edu/dataset) first.

## Training
Train the ACME model:GPU
```
CUDA_VISIBLE_DEVICES=0 python train.py 
```
We did the experiments with batch size 64, which takes about 12 GB memory.


## Model for Testing
Test the model:
```
CUDA_VISIBLE_DEVICES=0 python test.py
```
Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1svtpy-sD4pcaFfLGQNGaPIVjrKr-lhsT?usp=sharing). 
