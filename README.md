# EC601_Team12

Bonjour â tous! 😁

This is the repository of our first homework assignment for **EC601: Product Design in ECE (Fall 2017).**

## Team Members

[**Minghe Ren**](https://github.com/rmhsawyer): sawyermh@bu.edu

[**Shuang Zhao**](https://github.com/ShuangZhao95): zs1995@bu.edu

[**Xin Li**](https://github.com/helloimlixin): bulixin@bu.edu

## Assignment Instructions

### Objectives

1️⃣ Agile Project Management

2️⃣ Code Management

3️⃣ Team Communications

### Specifications

#### Agile Project Management Using Trello

✅ Create accounts in Trello

✅ Open a new project: [**EC601**](https://trello.com/b/kt7cNDNs/ec601)

✅ Assign tasks to each team member with respective task durations

✅

✅

#### Code Management Using Github

✅ Create Accounts in Github

✅

✅

✅

✅

#### Team Communicatons Using Slack
  
✅ Create a team in Slack

✅ Communicate inside team using Slack


## Project Description

### Adversarially trained ImageNet models

Pre-trained ImageNet models from the following papers:

* [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)
* [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204)

### Contact
Originated from:
Author: Alexey Kurakin,
github: [AlexeyKurakin](https://github.com/AlexeyKurakin)

Revised from:Author: Minghe Ren, Shuang Zhao, Xin Li,
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
#     either "inception_v3" or "inception_resnet_v2"
# ${CHECKPOINT_PATH} - path to model checkpoint
# ${DATASET_DIR} - directory with ImageNet test set
# ${ADV_METHOD} - which method to use to generate adversarial images,
#   supported method:
#     "none" - use clean images from the dataset
#     "stepll" - one step towards least likely class method (StepLL),
#         see https://arxiv.org/abs/1611.01236 for details
#     "stepllnoise" - RAND+StepLL method from https://arxiv.org/abs/1705.07204
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

This will begin training a fresh model.  If and when the model becomes
sufficiently well-trained, it will reset the Eve model multiple times
and retrain it from scratch, outputting the accuracy thus obtained
in each run.

## Model differences from the paper

The model has been simplified slightly from the one described in
the paper - the convolutional layer width was reduced by a factor
of two.  In the version in the paper, there was a nonlinear unit
after the fully-connected layer;  that nonlinear has been removed
here.  These changes improve the robustness of training.  The
initializer for the convolution layers has switched to the
tf.contrib.layers default of xavier_initializer instead of
a simpler truncated_normal.

## Contact information

This model repository is maintained by David G. Andersen
([dave-andersen](https://github.com/dave-andersen)).



# NeuralGPU
Code for the Neural GPU model described in http://arxiv.org/abs/1511.08228.
The extended version was described in https://arxiv.org/abs/1610.08613.

Requirements:
* TensorFlow (see tensorflow.org for how to install)

The model can be trained on the following algorithmic tasks:

* `sort` - Sort a symbol list
* `kvsort` - Sort symbol keys in dictionary
* `id` - Return the same symbol list
* `rev` - Reverse a symbol list
* `rev2` - Reverse a symbol dictionary by key
* `incr` - Add one to a symbol value
* `add` - Long decimal addition
* `left` - First symbol in list
* `right` - Last symbol in list
* `left-shift` - Left shift a symbol list
* `right-shift` - Right shift a symbol list
* `bmul` - Long binary multiplication
* `mul` - Long decimal multiplication
* `dup` - Duplicate a symbol list with padding
* `badd` - Long binary addition
* `qadd` - Long quaternary addition
* `search` - Search for symbol key in dictionary

It can also be trained on the WMT English-French translation task:

* `wmt` - WMT English-French translation (data will be downloaded)

The value range for symbols are defined by the `vocab_size` flag.
In particular, the values are in the range `vocab_size - 1`.
So if you set `--vocab_size=16` (the default) then `--problem=rev`
will be reversing lists of 15 symbols, and `--problem=id` will be identity
on a list of up to 15 symbols.


To train the model on the binary multiplication task run:

```
python neural_gpu_trainer.py --problem=bmul
```

This trains the Extended Neural GPU, to train the original model run:

```
python neural_gpu_trainer.py --problem=bmul --beam_size=0
```

While training, interim / checkpoint model parameters will be
written to `/tmp/neural_gpu/`.

Once the amount of error gets down to what you're comfortable
with, hit `Ctrl-C` to stop the training process. The latest
model parameters will be in `/tmp/neural_gpu/neural_gpu.ckpt-<step>`
and used on any subsequent run.

To evaluate a trained model on how well it decodes run:

```
python neural_gpu_trainer.py --problem=bmul --mode=1
```

To interact with a model (experimental, see code) run:

```
python neural_gpu_trainer.py --problem=bmul --mode=2
```

To train on WMT data, set a larger --nmaps and --vocab_size and avoid curriculum:

```
python neural_gpu_trainer.py --problem=wmt --vocab_size=32768 --nmaps=256
  --vec_size=256 --curriculum_seq=1.0 --max_length=60 --data_dir ~/wmt
```

With less memory, try lower batch size, e.g. `--batch_size=4`. With more GPUs
in your system, there will be a batch on every GPU so you can run larger models.
For example, `--batch_size=4 --num_gpus=4 --nmaps=512 --vec_size=512` will
run a large model (512-size) on 4 GPUs, with effective batches of 4*4=16.

Maintained by Lukasz Kaiser (lukaszkaiser)

# Adversarial Text Classification

Code for [*Adversarial Training Methods for Semi-Supervised Text Classification*](https://arxiv.org/abs/1605.07725) and [*Semi-Supervised Sequence Learning*](https://arxiv.org/abs/1511.01432).

## Requirements

* Bazel ([install](https://bazel.build/versions/master/docs/install.html))
* TensorFlow >= v1.1

## End-to-end IMDB Sentiment Classification

### Fetch data

```
$ wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
    -O /tmp/imdb.tar.gz
$ tar -xf /tmp/imdb.tar.gz -C /tmp
```

The directory `/tmp/aclImdb` contains the raw IMDB data.

### Generate vocabulary

```
$ IMDB_DATA_DIR=/tmp/imdb
$ bazel run data:gen_vocab -- \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/aclImdb \
    --lowercase=False
```

Vocabulary and frequency files will be generated in `$IMDB_DATA_DIR`.

###  Generate training, validation, and test data

```
$ bazel run data:gen_data -- \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/aclImdb \
    --lowercase=False \
    --label_gain=False
```

`$IMDB_DATA_DIR` contains TFRecords files.

### Pretrain IMDB Language Model

```
$ PRETRAIN_DIR=/tmp/models/imdb_pretrain
$ bazel run :pretrain -- \
    --train_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=86934 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --num_candidate_samples=1024 \
    --batch_size=256 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.9999 \
    --max_steps=100000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings
```

`$PRETRAIN_DIR` contains checkpoints of the pretrained language model.

### Train classifier

Most flags stay the same, save for the removal of candidate sampling and the
addition of `pretrained_model_dir`, from which the classifier will load the
pretrained embedding and LSTM variables, and flags related to adversarial
training and classification.

```
$ TRAIN_DIR=/tmp/models/imdb_classify
$ bazel run :train_classifier -- \
    --train_dir=$TRAIN_DIR \
    --pretrained_model_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=86934 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --cl_num_layers=1 \
    --cl_hidden_size=30 \
    --batch_size=64 \
    --learning_rate=0.0005 \
    --learning_rate_decay_factor=0.9998 \
    --max_steps=15000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings \
    --adv_training_method=vat \
    --perturb_norm_length=5.0
```

### Evaluate on test data

```
$ EVAL_DIR=/tmp/models/imdb_eval
$ bazel run :evaluate -- \
    --eval_dir=$EVAL_DIR \
    --checkpoint_dir=$TRAIN_DIR \
    --eval_data=test \
    --run_once \
    --num_examples=25000 \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=86934 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --batch_size=256 \
    --num_timesteps=400 \
    --normalize_embeddings
```

## Code Overview

The main entry points are the binaries listed below. Each training binary builds
a `VatxtModel`, defined in `graphs.py`, which in turn uses graph building blocks
defined in `inputs.py` (defines input data reading and parsing), `layers.py`
(defines core model components), and `adversarial_losses.py` (defines
adversarial training losses). The training loop itself is defined in
`train_utils.py`.

### Binaries

*   Pretraining: `pretrain.py`
*   Classifier Training: `train_classifier.py`
*   Evaluation: `evaluate.py`

### Command-Line Flags

Flags related to distributed training and the training loop itself are defined
in [`train_utils.py`](https://github.com/tensorflow/models/tree/master/adversarial_text/train_utils.py).

Flags related to model hyperparameters are defined in [`graphs.py`](https://github.com/tensorflow/models/tree/master/adversarial_text/graphs.py).

Flags related to adversarial training are defined in [`adversarial_losses.py`](https://github.com/tensorflow/models/tree/master/adversarial_text/adversarial_losses.py).

Flags particular to each job are defined in the main binary files.

### Data Generation

*   Vocabulary generation: [`gen_vocab.py`](https://github.com/tensorflow/models/tree/master/adversarial_text/data/gen_vocab.py)
*   Data generation: [`gen_data.py`](https://github.com/tensorflow/models/tree/master/adversarial_text/data/gen_data.py)

Command-line flags defined in [`document_generators.py`](https://github.com/tensorflow/models/tree/master/adversarial_text/data/document_generators.py)
control which dataset is processed and how.

## Contact for Issues

* Ryan Sepassi, @rsepassi
