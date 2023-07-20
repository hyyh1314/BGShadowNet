# Document Image Shadow Removal Guided by Color-Aware Background

## Requirements

+ python 3.6.13
+ pytorch 1.10.0
+ numpy==1.19.2
+ opencv_python_headless==4.5.4.60

##  TRAIN

1. You can obtain RDD dataset from [here](https://github.com/hyyh1314/RDD)
2. Modify the `config.yaml` to set your parameters
3. You can generate the csv file you need by configuring, running utils/make_dataset.py
4. Use extractBackground.py to construct the ground-truth background
5. Training CBENet

```python Train_CBENet.py ./configs/model=CBENet/config.yaml```

5. Training BGShadowNet

```python Train_BGShadowNet.py ./configs/model=BGShadowNet/config.yaml```

6. Pretrained model

Download the [pretrained model](https://drive.google.com/file/d/1zHolTv-fj_RsSUD1Je8-ymcnkuSBi8QR/view?usp=sharing) trained on RDD dataset. 
## TEST
 ```python test.py```

## Cite

```@InProceedings{Zhu_2022_CVPR,
  @InProceedings{Zhang_2023_CVPR,
  author  = {Ling Zhang,Yinghao He,Qing Zhang,Zheng Liu,Xiaolong Zhang,Chunxia Xiao},
  title   = {Document Image Shadow Removal Guided by Color-Aware Background.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month   = {June},
  year   = {2023}}
```

