# HiGCIN
Pytorch implementation of [HiGCIN: Hierarchical Graph-based Cross Inference Network for Group Activity Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9241410). (IEEE T-PAMI, 2020)



## Get started
### Prerequisite
Our approach is tested on only Ubuntu with GPU and it needs at least 16G GPU memory. The neccseearay packages can be install by following commonds:
```
conda create -n HiGCIN python=3.6
conda activate HiGCIN
pip install cmake dlib scikit-image sklearn h5py
pip install torch torchvision
```
### Preprocess datasets
Download two datasets (*i.e.*, [Volleyball Dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and [Collective Activity Dataset](http://www.eecs.umich.edu/vision/activity-dataset.html)) and unzip them to **'./dataset/VD/videos'** and **'./dataset/CAD/videos'**, respectively. Then run the following command:
```
bash pre_script.sh 'VD'
bash pre_script.sh 'CAD'
```

Alternatively, you can also direct download the personal tracklets from [here](https://note.youdao.com/) and put them in **'./dataset/VD/imgs'** and **'./dataset/CAD/imgs'**, respectively.
### Train a Standard Model from Scratch
```
bash traintest_script.sh
```

## Citation
If you wish to refer to the results of HiGCIN, please use the following BibTeX entry.

```
@article{yan2020higcin,
  title={HiGCIN: Hierarchical Graph-based Cross Inference Network for Group Activity Recognition},
  author={Yan, Rui and Xie, Lingxi and Tang, Jinhui and Shu, Xiangbo and Tian, Qi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  doi={10.1109/TPAMI.2020.3034233}
}
```


## Update Records
12/3/2020: Only Volleyball Dataset is supported now.

## Todo
- Support Collective Activity Dataset


## Acknowledgments
Thanks to the pytorch version implementation of Non-Local from [https://github.com/AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)

## Contact Information
Feel free to create a pull request or contact me by Email = ["ruiyan", at, "njust", dot, "edu", dot, "cn"], if you find any bugs. For further information about me, welcome to my [homepage](https://ruiyan1995.github.io/).