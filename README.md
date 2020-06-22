#RCDNet: A Model-driven Deep Neural Network  for Single Image Rain Removal (CVPR2020)
[Hong Wang](https://hongwang01.github.io/), Qi Xie, Qian Zhao, and [Deyu Meng](http://gr.xjtu.edu.cn/web/dymeng)  

[[PDF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf) [[Supplementary Material]](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Wang_A_Model-Driven_Deep_CVPR_2020_supplemental.pdf)

## Abstract
Deep learning (DL) methods have achieved state-of-the-art performance in the task of single image rain removal. Most of current DL architectures, however, are still lack of sufficient interpretability and not fully integrated with physical structures inside general rain streaks. To this issue, in this paper, we propose a model-driven deep neural network for the task, with fully interpretable network structures. Specifically, based on the convolutional dictionary learning mechanism for representing rain, we propose a novel single image deraining model and utilize the proximal gradient descent technique to design an iterative algorithm only containing simple operators for solving the model. Such a simple implementation scheme facilitates us to unfold it into a new deep network architecture, called rain convolutional dictionary network (RCDNet), with almost every network module one-to-one corresponding to each operation involved in the algorithm. By end-to-end training the proposed RCDNet, all the rain kernels and proximal operators can be automatically extracted, faithfully characterizing the features of both rain and clean background layers, and thus naturally lead to its better deraining performance, especially in real scenarios. Comprehensive experiments substantiate the superiority of the proposed network, especially its well generality to diverse testing scenarios and good interpretability for all its modules, as compared with state-of-the-arts both visually and quantitatively.

## Citation
```
@InProceedings{Wang_2020_CVPR,  
author = {Wang, Hong and Xie, Qi and Zhao, Qian and Meng, Deyu},  
title = {A Model-Driven Deep Neural Network for Single Image Rain Removal},  
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
month = {June},  
year = {2020}  
}
```
## Requirements

* Linux Platform
* NVIDIA GPU + CUDA CuDNN (CUDA 10.0)
* PyTorch == 0.4.1 (1.0.x and higher version needs revise the dataloader.py in folder "../src/", refer to "Friendly Tips" for details)
* Python3.6
* imageio2.5
* numpy1.14
* opencv-python
* scikit-image0.13.0
* tqdm4.32.2

## Dataset Descriptions 
### Synthetic datasets
* Rain100L: 200 training pairs and 100 testing pairs
* Rain100H: 1800 training pairs and 100 testing pairs
* Rain1400: 12600 training pairs and 1400 testing pairs
* Rain12: 12 testing pairs
### SPA-Data
* 638492 training pairs and 1000 testing pairs
### Internet-Data
* 147 unlabeled rainy images from [SIRR](https://github.com/wwzjer/Semi-supervised-IRR/tree/master/data/rainy_image_dataset/real_input)

**More detailed explantions refer to [the single image part in the Survey](https://github.com/hongwang01/Video-and-Single-Image-Deraining)*

## Training
###  For Synthetic Dataset 
*taking training Rain100L as an example*:
1. Download Rain100L  (including training set and testing set) from the  [[NetDisk]](https://pan.baidu.com/s/1yV4ih7C4Xg0iazqSBB-U1Q) (pwd:uz8h) and put them into the folder "./RCDNet_code/for_syn/data",  then the content is just like:

"./for_syn/data/train/small/rain/\*.png"

"./for_syn/data/train/small/norain/\*.png"

 "./for_syn/data/test/small/rain/\*.png"
 
"./for_syn/data/test/small/norain/\*.png"

2.  Begining training:
```
$ cd ./RCDNet_code/for_syn/src/ 
$ python main.py  --save RCDNet_syn --model RCDNet --scale 2 --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 25 --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-200/1-100 --loss 1*MSE --save_models 
```
**Note that: --data_range 1-#training pairs/1-#testing pairs.  The command above is set based on Rain100L (200 training paris and 100 testing pairs).*

### For SPA-Data
1. Download the training set from [[SPANet]](_https://stevewongv.github.io/derain-project.html) and testing set from the  [[NetDisk]](https://pan.baidu.com/s/1yV4ih7C4Xg0iazqSBB-U1Q) (pwd:uz8h) and put them into the folder "./RCDNet_code/for_spa/data",  then the content is just like:

"./for_spa/data/real_world"

"./for_spa/data/real_world_gt"

"./for_spa/data/real_world.txt"

 "./for_spa/data/test/small/rain/\*.png"
 
"./for_spa/data/test/small/norain/\*.png"

**Note that:  download the training set  from [[SPANet]](https://stevewongv.github.io/derain-project.html) ,  which includes three files: "real_world"， "real_world_gt", and "real_world.txt"*


2.  Begining training:
```
$ cd ./RCDNet_code/for_spa/src/ 
$ python main.py  --save RCDNet_spa --model RCDNet --scale 2 --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 25 --gamma 0.2  --num_M 32 --num_Z 32 --data_range 1-638492/1-1000 --loss 1*MSE --save_models 
```

## Testing
### For Synthetic Dataset
```
$ cd ./RCDNet_code/for_syn/src/
$ python main.py --data_test RainHeavyTest  --ext img --scale 2  --data_range 1-200/1-100 --pre_train ../experiment/RCDNet_syn/model/model_best.pt --model RCDNet --test_only --save_results --save RCDNet_test
```
**Note that: --data_range  1-#training pairs/1-#testing pairs. The command (-- data_range 1-200/1-100) above is set based on Rain100L (200 training paris and 100 testing pairs).*

The derained results are saved in the folder "./for_syn/experiment/RCDNet_test/results/", where the image name "norain-*LR.png"  , "norain-*HR.png" , "norain-*SR.png" means rainy image, groundtruth,  and restored background, respectively. 

### For SPA-Data
```
$ cd ./RCDNet_code/for_spa/src/
$ python main.py --data_test RainHeavyTest  --ext img --scale 2  --data_range 1-638492/1-1000 --pre_train ../experiment/RCDNet_spa/model/model_best.pt --model RCDNet --test_only --save_results --save RCDNet_test
```
The derained results are saved in the folder "./for_spa/experiment/RCDNet_test/results/"
## Pretrained Model
See folder "Pretrained_model" 

## Derained Results
If needed, please download from  [[NetDisk]](https://pan.baidu.com/s/1IU5hT8Vjcyf8wNYuQD5rPw)(pwd:e2xo) 

1) *for synthetic model*:  put the pretrained model into the folder "./for_syn/experiment/RCDNet_syn/model/",  and then the content is just like: "../for_syn/experiment/RCDNet_syn/model/model_best.pt"

2) *for spa mode*l:  put the pretrained model into the folder "./for_spa/experiment/RCDNet_spa/model/",  and then the content is just like: "../for_spa/experiment/RCDNet_spa/model/model_best.pt"



## Performance Evaluation

All PSNR and SSIM results are computed by using  this [Matlab code](https://github.com/hongwang01/RCDNet/tree/master/Performance_evaluation), based on Y channal.

## Acknowledgement 
Code borrows from [JORDER_E](https://github.com/flyywh/JORDER-E-Deep-Image-Deraining-TPAMI-2019-Journal) by [Wenhan Yang](https://github.com/flyywh). Thanks for sharing !

## Friendly Tips
1. If higher version is needed, please replace the files "dataloader.py"  and "trainer.py" in the folder "./RCDNet_code/../src/" with the  corresponding ones in  the folder "./pytorch1.0+/../src/" .
2. More general and friendly code framework for RCDNet will be coming !

## Updating logs
2020.6.10   Release code
2020.6.22   Fix the bug in the folder "../src/model/rcdnet/" and also release higher version with the folder "./pytorch1.0+"

## Contact
If you have any question, please feel free to concat Hong Wang (Email: hongwang01@stu.xjtu.edu.cn)
