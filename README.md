# EC601_team12
# Adversarially trained ImageNet models

Pre-trained ImageNet models from the following papers:

* [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)
* [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204)

## Contact
Originated from:
Author: Alexey Kurakin,
github: [AlexeyKurakin](https://github.com/AlexeyKurakin)

Revised from:
Author: Minghe Ren, Shuang Zhao, Xin Li,
github:https://github.com/rmhsawyer/EC601-Project1/edit/Minghe-Ren/README.md



## Pre-requesites and installation

Ensure that you have installed TensorFlow 1.1 or greater
([instructions](https://www.tensorflow.org/install/)).

You also need copy of ImageNet dataset if you want to run provided example.
Follow
[Preparing the dataset](https://github.com/tensorflow/models/tree/master/slim#Data)
instructions in TF-Slim library to get and preprocess ImageNet data.

## Available models

Following pre-trained models are available:

Network Architecture | Adversarial training | Checkpoint
---------------------|----------------------|----------------
Inception v3 | Step L.L. | [adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz)
Inception v3 | Step L.L. on ensemble of 3 models | [ens3_adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz)
Inception v3 | Step L.L. on ensemble of 4 models| [ens4_adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz)
Inception ResNet v2 | Step L.L. on ensemble of 3 models | [ens_adv_inception_resnet_v2_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz)

All checkpoints are compatible with
[TF-Slim](https://github.com/tensorflow/models/tree/master/slim)
implementation of Inception v3 and Inception Resnet v2.

## How to evaluate models on ImageNet test data

Python script `eval_on_adversarial.py` allow you to evaluate provided models
on white-box adversarial examples generated from ImageNet test set.

Usage is following:

```bash
# ${MODEL_NAME} - type of network architecture,
#     either "inception_v3" or "inception_resnet_v2"
# ${CHECKPOINT_PATH} - path to model checkpoint
# ${DATASET_DIR} - directory with ImageNet test set
# ${ADV_METHOD} - which method to use to generate adversarial images,
#   supported method:
#     "none" - use clean images from the dataset
#     "stepll" - one step towards least likely class method (StepLL),
#         see https://arxiv.org/abs/1611.01236 for details
#     "stepllnoise" - RAND+StepLL method from https://arxiv.org/abs/1705.07204
# ${ADV_EPS} - size of adversarial perturbation, ignored when method is none
python eval_on_adversarial.py \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --dataset_dir=${DATASET_DIR} \
  --batch_size=50 \
  --adversarial_method=${ADV_METHOD} \
  --adversarial_eps=${ADV_EPS}
```

Below is an example how to evaluate one of the models on RAND+StepLL adversarial
examples:

```bash
# Download checkpoint
CHECKPOINT_DIR=/tmp/checkpoints
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
mv ens_adv_inception_resnet_v2.ckpt* ${CHECKPOINT_DIR}
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Run evaluation
python eval_on_adversarial.py \
  --model_name=inception_v3 \
  --checkpoint_path=${CHECKPOINT_DIR}/ens_adv_inception_resnet_v2.ckpt \
  --dataset_dir=${DATASET_DIR} \
  --batch_size=50 \
  --adversarial_method=stepllnoise \
  --adversarial_eps=16
```




# Learning to Protect Communications with Adversarial Neural Cryptography

This is a slightly-updated model used for the paper
["Learning to Protect Communications with Adversarial Neural
Cryptography"](https://arxiv.org/abs/1610.06918).

> We ask whether neural networks can learn to use secret keys to protect
> information from other neural networks. Specifically, we focus on ensuring
> confidentiality properties in a multiagent system, and we specify those
> properties in terms of an adversary. Thus, a system may consist of neural
> networks named Alice and Bob, and we aim to limit what a third neural
> network named Eve learns from eavesdropping on the communication between
> Alice and Bob. We do not prescribe specific cryptographic algorithms to
> these neural networks; instead, we train end-to-end, adversarially.
> We demonstrate that the neural networks can learn how to perform forms of
> encryption and decryption, and also how to apply these operations
> selectively in order to meet confidentiality goals.

This code allows you to train an encoder/decoder/adversary triplet
and evaluate their effectiveness on randomly generated input and key
pairs.

## Prerequisites

The only software requirements for running the encoder and decoder is having
Tensorflow installed.

Requires Tensorflow r0.12 or later.

## Training and evaluating

After installing TensorFlow and ensuring that your paths are configured
appropriately:

```
python train_eval.py
```

This will begin training a fresh model.  If and when the model becomes
sufficiently well-trained, it will reset the Eve model multiple times
and retrain it from scratch, outputting the accuracy thus obtained
in each run.

## Model differences from the paper

The model has been simplified slightly from the one described in
the paper - the convolutional layer width was reduced by a factor
of two.  In the version in the paper, there was a nonlinear unit
after the fully-connected layer;  that nonlinear has been removed
here.  These changes improve the robustness of training.  The
initializer for the convolution layers has switched to the
tf.contrib.layers default of xavier_initializer instead of
a simpler truncated_normal.

## Contact information

This model repository is maintained by David G. Andersen
([dave-andersen](https://github.com/dave-andersen)).
