# RaftMLP

RaftMLP: How Much Can Be Done Without Attention and with Less Spatial Locality?

By Yuki Tatsunami and Masato Taki (Rikkyo University)

[[arxiv](https://arxiv.org/abs/2108.04384v2)]

## Abstract

For the past ten years, CNN has reigned supreme in the world of computer vision, but recently, Transformer has been on the rise. However, the quadratic computational cost of self-attention has become a serious problem in practice applications. There has been much research on architectures without CNN and self-attention in this context. In particular, MLP-Mixer is a simple architecture designed using MLPs and hit an accuracy comparable to the Vision Transformer. However, the only inductive bias in this architecture is the embedding of tokens. This leaves open the possibility of incorporating a non-convolutional (or non-local) inductive bias into the architecture, so we used two simple ideas to incorporate inductive bias into the MLP-Mixer while taking advantage of its ability to capture global correlations. A way is to divide the token-mixing block vertically and horizontally. Another way is to make spatial correlations denser among some channels of token-mixing. With this approach, we were able to improve the accuracy of the MLP-Mixer while reducing its parameters and computational complexity. The small model that is RaftMLP-S is comparable to the state-of-the-art global MLP-based model in terms of parameters and efficiency per calculation. In addition, we tackled the problem of fixed input image resolution for global MLP-based models by utilizing bicubic interpolation. We demonstrated that these models could be applied as the backbone of architectures for downstream tasks such as object detection. However, it did not have significant performance and mentioned the need for MLP-specific architectures for downstream tasks for global MLP-based models.

## About Environment

Our base is PyTorch, Torchvision, and Ignite. We use mmdetection and mmsegmentation for object detection and semantic segmentation. We also use ClearML, AWS, etc., for experiment management.


We also use Docker for our environment, and with Docker and NVIDIA Container Toolkit installed, we can build a runtime environment at the ready.

## Require

+ NVIDIA Driver
+ Docker(19.03+)
+ Docker Compose(1.28.0+)
+ NVIDIA Container Toolkit

## Prepare

### clearml.conf

Please copy clearml.conf.sample, you can easily create clearml.conf. Unless you have a Clear ML account, you should use the account. Next, you obtain the access key and secret key of the service. Let's write them on clearml.conf. If you don't have an AWS account, you will need one. Then, create an IAM user and an S3 bucket, and grant the IAM user a policy that allows you to read and write objects to the bucket you created. Include the access key and secret key of the IAM user you created and the region of the bucket you made in your clearml.conf.

### docker-compose.yml

Please copy docker-compose.yml.sample to docker-compose.yml. Change the path/to/datasets in the volumes section to an appropriate directory where the datasets are stored. You can set device_ids on your environment. If you train semantic segmentation models or object detection models, you should set WANDB_API_KEY.

## Datasets

Except for ImageNet, our codes automatically download datasets, but we recommend downloading them beforehand. Datasets need to be placed in the location set in the datasets directory in docker-compose.yml.

### ImageNet1k

Please go to [URL](https://image-net.org/) and register on the site. Then you can download ImageNet1k dataset. You should place it under path/to/datasets with the following structure.

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### CIFAR10

No problem, just let the code download automatically.
[URL](https://www.cs.toronto.edu/~kriz/cifar.html)

### CIFAR100

No problem, just let the code download automatically.
[URL](https://www.cs.toronto.edu/~kriz/cifar.html)

### Oxford 102 Flowers

No problem, just let the code download automatically.
[URL](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

### Stanford Cars

You should place it under path/to/datasets with the following structure.

```
│stanford_cars/
├──cars_train/
│  ├── 00001.jpg
│  ├── 00002.jpg
│  ├── ......
├──cars_test/
│  ├── 00001.jpg
│  ├── 00002.jpg
│  ├── ......
├──devkit/
│  ├── cars_meta.mat
│  ├── cars_test_annos.mat
│  ├── cars_train_annos.mat
│  ├── eval_train.m
│  ├── README.txt
│  ├── train_perfect_preds.txt
├──cars_test_annos_withlabels.matcars_test_annos_withlabels.mat
```

[URL](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

### iNaturalist18

You should place it under path/to/datasets with the following structure.


```
│i_naturalist_18/
├──train_val2018/
│  ├──Actinopterygii/
│  │  ├──2229/
│  │  │  ├── 014a31153ac74bf87f1f730480e4a27a.jpg
│  │  │  ├── 037d062cc1b8a85821449d2cdeca7749.jpg
│  │  │  ├── ......
│  │  ├── ......
│  ├── ......
├──train2018.json
├──val2018.json
```

[URL](https://github.com/visipedia/inat_comp/tree/2018)

### iNaturalist19

You should place it under path/to/datasets with the following structure.


```
│i_naturalist_19/
├──train_val2019/
│  ├──Amphibians/
│  │  ├──153/
│  │  │  ├── 0042d05b4ffbd5a1ce2fc56513a7777e.jpg
│  │  │  ├── 006f69e838b87cfff3d12120795c4ada.jpg
│  │  │  ├── ......
│  │  ├── ......
│  ├── ......
├──train2019.json
├──val2019.json
```

[URL](https://github.com/visipedia/inat_comp/tree/2019)

### MS COCO

You should place it under path/to/datasets with the following structure.

```
│coco/
├──train2017/
│  ├── 000000000009.jpg
│  ├── 000000000025.jpg
│  ├── ......
├──val2017/
│  ├── 000000000139.jpg
│  ├── 000000000285.jpg
│  ├── ......
├──annotations/
│  ├── captions_train2017.json
│  ├── captions_val2017.json
│  ├── instances_train2017.json
│  ├── instances_val2017.json
│  ├── person_keypoints_train2017.json
│  ├── person_keypoints_val2017.json
```

[URL](https://cocodataset.org/)

### ADE20K

In order for you to download the ADE20k dataset, you have to register at this [site](https://groups.csail.mit.edu/vision/datasets/ADE20K/) and get approved. Once downloaded the dataset, place it so that it has the following structure.

```
│ade/
├──ADEChallengeData2016/
│  ├──annotations/
│  │  ├──training/
│  │  │  ├── ADE_train_00000001.png
│  │  │  ├── ADE_train_00000002.png
│  │  │  ├── ......
│  │  ├──validation/
│  │  │  ├── ADE_val_00000001.png
│  │  │  ├── ADE_val_00000002.png
│  │  │  ├── ......
│  ├──images/
│  │  ├──training/
│  │  │  ├── ADE_train_00000001.jpg
│  │  │  ├── ADE_train_00000002.jpg
│  │  │  ├── ......
│  │  ├──validation/
│  │  │  ├── ADE_val_00000001.jpg
│  │  │  ├── ADE_val_00000002.jpg
│  │  │  ├── ......
│  │  ├──
│  ├──objectInfo150.txt
│  ├──sceneCategories.txt
```

## ImageNet1k

configs/settings are available. Each of the training conducted in Subsection 4.1 can be performed in the following commands.

```shell
docker run trainer python run.py settings=imagenet-raft-mlp-cross-mlp-emb-s
docker run trainer python run.py settings=imagenet-raft-mlp-cross-mlp-emb-m
docker run trainer python run.py settings=imagenet-raft-mlp-cross-mlp-emb-l
```
The ablation study for channel rafts in subsection 4.2 ran the following commands.


## Abrasion Study

```shell
docker run trainer python run.py settings=imagenet-org-mixer
docker run trainer python run.py settings=imagenet-raft-mlp-r-1
docker run trainer python run.py settings=imagenet-raft-mlp-r-2
docker run trainer python run.py settings=imagenet-raft-mlp
```

The abrasion study for multi-scale patch embedding in subsection 4.2 ran the following commands.

```shell
docker run trainer python run.py settings=imagenet-raft-mlp-cross-mlp-emb-m
docker run trainer python run.py settings=imagenet-raft-mlp-hierarchy-m
```

## Transfer Learning

```shell
docker run trainer python run.py settings=finetune/cars-org-mixer.yaml
docker run trainer python run.py settings=finetune/cars-raft-mlp-cross-mlp-emb-s.yaml
docker run trainer python run.py settings=finetune/cars-raft-mlp-cross-mlp-emb-m.yaml
docker run trainer python run.py settings=finetune/cars-raft-mlp-cross-mlp-emb-l.yaml
docker run trainer python run.py settings=finetune/cifar10-org-mixer.yaml
docker run trainer python run.py settings=finetune/cifar10-raft-mlp-cross-mlp-emb-s.yaml
docker run trainer python run.py settings=finetune/cifar10-raft-mlp-cross-mlp-emb-m.yaml
docker run trainer python run.py settings=finetune/cifar10-raft-mlp-cross-mlp-emb-l.yaml
docker run trainer python run.py settings=finetune/cifar100-org-mixer.yaml
docker run trainer python run.py settings=finetune/cifar100-raft-mlp-cross-mlp-emb-s.yaml
docker run trainer python run.py settings=finetune/cifar100-raft-mlp-cross-mlp-emb-m.yaml
docker run trainer python run.py settings=finetune/cifar100-raft-mlp-cross-mlp-emb-l.yaml
docker run trainer python run.py settings=finetune/flowers102-org-mixer.yaml
docker run trainer python run.py settings=finetune/flowers102-raft-mlp-cross-mlp-emb-s.yaml
docker run trainer python run.py settings=finetune/flowers102-raft-mlp-cross-mlp-emb-m.yaml
docker run trainer python run.py settings=finetune/flowers102-raft-mlp-cross-mlp-emb-l.yaml
docker run trainer python run.py settings=finetune/inat18-org-mixer.yaml
docker run trainer python run.py settings=finetune/inat18-raft-mlp-cross-mlp-emb-s.yaml
docker run trainer python run.py settings=finetune/inat18-raft-mlp-cross-mlp-emb-m.yaml
docker run trainer python run.py settings=finetune/inat18-raft-mlp-cross-mlp-emb-l.yaml
docker run trainer python run.py settings=finetune/inat19-org-mixer.yaml
docker run trainer python run.py settings=finetune/inat19-raft-mlp-cross-mlp-emb-s.yaml
docker run trainer python run.py settings=finetune/inat19-raft-mlp-cross-mlp-emb-m.yaml
docker run trainer python run.py settings=finetune/inat19-raft-mlp-cross-mlp-emb-l.yaml
```

## Object Detection

The weights already trained by ImageNet should be placed in the following path.

```
path/to/datasets/weights/imagenet-raft-mlp-cross-mlp-emb-s/last_model_0.pt
path/to/datasets/weights/imagenet-raft-mlp-cross-mlp-emb-l/last_model_0.pt
path/to/datasets/weights/imagenet-raft-mlp-cross-mlp-emb-m/last_model_0.pt
path/to/datasets/weights/imagenet-org-mixer/last_model_0.pt
```

Please execute the following commands.

```shell
docker run trainer bash ./detection.sh configs/detection/maskrcnn_org_mixer_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./detection.sh configs/detection/maskrcnn_raftmlp_l_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./detection.sh configs/detection/maskrcnn_raftmlp_m_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./detection.sh configs/detection/maskrcnn_raftmlp_s_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./detection.sh configs/detection/retinanet_org_mixer_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./detection.sh configs/detection/retinanet_raftmlp_l_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./detection.sh configs/detection/retinanet_raftmlp_m_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./detection.sh configs/detection/retinanet_raftmlp_s_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
```

## Semantic Segmentation

As with object detection, the following should be executed after placing the weight files in advance.

```shell
docker run trainer bash ./segmentation.sh configs/segmentation/fpn_org_mixer_512x512_40k_ade20k.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./segmentation.sh configs/segmentation/fpn_raftmlp_s_512x512_40k_ade20k.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./segmentation.sh configs/segmentation/fpn_raftmlp_m_512x512_40k_ade20k.py 8 --seed=42 --deterministic --gpus=8
docker run trainer bash ./segmentation.sh configs/segmentation/fpn_raftmlp_l_512x512_40k_ade20k.py 8 --seed=42 --deterministic --gpus=8
```

## Reference

```
@misc{tatsunami2021raftmlp,
  title={RaftMLP: How Much Can Be Done Without Attention and with Less Spatial Locality?},
  author={Yuki Tatsunami and Masato Taki},
  year={2021}
  eprint={2108.04384},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License
Apache 2.0 Licensed.
