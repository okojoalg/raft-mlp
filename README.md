# RaftMLP

This repository contains the training code for RaftMLP.

## Require

+ cuda toolkit(10.2+)
+ cuDNN
+ nvidia-container-runtime
+ docker-compose(1.28.0+)

## Prepare

First, please prepare clearml.conf. If you copy clearml.conf.sample, you can easily create it. Unless you have a Clear ML account, you should the account. Next, you obtain the access key and secret key of the service. Let's write them on clearml.conf. If you don't have an AWS account, you will need one. Then, create an IAM user and an S3 bucket, and grant the IAM user a policy that allows you to read and write objects to the bucket you created. Include the access key and secret key of the IAM user you created and the region of the bucket you created in your clearml.conf.
You will need to prepare docker-compose.yml as well, copy docker-compose.yml.sample and create it. Specify the id of the gpu to be used in the device_ids field. Please also put each archive file of imagenet to ./data/imagenet. In the command field, specify the name of the configuration file to be used after the run.py argument settings=. You can also replace run.py with summary.py to display a summary of the model.

## Train

Please execute the following command:

```docker-compose up -d```

This will build the docker image, start the container, and run the training code on the container. The results and models during execution are uploaded to ClearML by the process and can be viewed from the ClearML web app.

≈bash ./segmentation.sh configs/segmentation/fpn_raftmlp_l_512x512_40k_ade20k.py 1 --seed=42 --deterministic --gpus=1
bash ./detection.sh configs/detection/retinanet_org_mixer_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1
bash ./detection.sh configs/detection/maskrcnn_raftmlp_l_fpn_1x_coco.py 1 --seed=1 --deterministic --gpus=1

bash ./segmentation.sh configs/segmentation/fpn_org_mixer_512x512_40k_ade20k.py 8 --seed=42 --deterministic --gpus=8
bash ./segmentation.sh configs/segmentation/fpn_raftmlp_l_512x512_40k_ade20k.py 8 --seed=42 --deterministic --gpus=8
bash ./detection.sh configs/detection/retinanet_org_mixer_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8
bash ./detection.sh configs/detection/maskrcnn_raftmlp_l_fpn_1x_coco.py 8 --seed=42 --deterministic --gpus=8

bash ./segmentation.sh configs/segmentation/fpn_org_mixer_512x512_40k_ade20k.py 1 --seed=42 --deterministic --gpus=1
63,938,598
bash ./segmentation.sh configs/segmentation/fpn_raftmlp_s_512x512_40k_ade20k.py 1 --seed=42 --deterministic --gpus=1
13,605,006
bash ./segmentation.sh configs/segmentation/fpn_raftmlp_m_512x512_40k_ade20k.py 1 --seed=42 --deterministic --gpus=1
25,016,238
bash ./segmentation.sh configs/segmentation/fpn_raftmlp_l_512x512_40k_ade20k.py 1 --seed=42 --deterministic --gpus=1
39,637,262

bash ./detection.sh configs/detection/retinanet_org_mixer_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1 | grep "Total params";
70,331,524
bash ./detection.sh configs/detection/retinanet_raftmlp_s_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1 | grep "Total params";
19,588,332
bash ./detection.sh configs/detection/retinanet_raftmlp_m_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1 | grep "Total params";
31,581,196
bash ./detection.sh configs/detection/retinanet_raftmlp_l_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1 | grep "Total params";
46,783,852
bash ./detection.sh configs/detection/maskrcnn_org_mixer_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1 | grep "Total params";
79,839,360
bash ./detection.sh configs/detection/maskrcnn_raftmlp_s_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1 | grep "Total params";
29,505,768
bash ./detection.sh configs/detection/maskrcnn_raftmlp_m_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1 | grep "Total params";
40,917,000
bash ./detection.sh configs/detection/maskrcnn_raftmlp_l_fpn_1x_coco.py 1 --seed=42 --deterministic --gpus=1 | grep "Total params";
55,538,024