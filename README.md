# Pyramid Mixer

This repository contains the training code for Pyramid Mixer.

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
