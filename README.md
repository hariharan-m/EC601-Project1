-# EC601_team12
-#
-# Adversarially trained ImageNet models
-
-Pre-trained ImageNet models from the following papers:
-
-* [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)
-* [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204)
-
-## Contact
-Originated from:
-Author: Alexey Kurakin,
-github: [AlexeyKurakin](https://github.com/AlexeyKurakin)
-
-Revised from:Author: Minghe Ren, Shuang Zhao, Xin Li,
-github:https://github.com/rmhsawyer/EC601-Project1/edit/Minghe-Ren/README.md
-
-
-## Pre-requesites and installation
-
-Ensure that you have installed TensorFlow 1.1 or greater
-([instructions](https://www.tensorflow.org/install/)).
-
-You also need copy of ImageNet dataset if you want to run provided example.
-Follow
-[Preparing the dataset](https://github.com/tensorflow/models/tree/master/slim#Data)
-instructions in TF-Slim library to get and preprocess ImageNet data.
-
-## Available models
-
-Following pre-trained models are available:
-
-Network Architecture | Adversarial training | Checkpoint
----------------------|----------------------|----------------
-Inception v3 | Step L.L. | [adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz)
-Inception v3 | Step L.L. on ensemble of 3 models | [ens3_adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz)
-Inception v3 | Step L.L. on ensemble of 4 models| [ens4_adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz)
-Inception ResNet v2 | Step L.L. on ensemble of 3 models | [ens_adv_inception_resnet_v2_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz)
-
-All checkpoints are compatible with
-[TF-Slim](https://github.com/tensorflow/models/tree/master/slim)
-implementation of Inception v3 and Inception Resnet v2.
-
-## How to evaluate models on ImageNet test data
-
-Python script `eval_on_adversarial.py` allow you to evaluate provided models
-on white-box adversarial examples generated from ImageNet test set.
-
-Usage is following:
-
-```bash
-# ${MODEL_NAME} - type of network architecture,
-#     either "inception_v3" or "inception_resnet_v2"
-# ${CHECKPOINT_PATH} - path to model checkpoint
-# ${DATASET_DIR} - directory with ImageNet test set
-# ${ADV_METHOD} - which method to use to generate adversarial images,
-#   supported method:
-#     "none" - use clean images from the dataset
-#     "stepll" - one step towards least likely class method (StepLL),
-#         see https://arxiv.org/abs/1611.01236 for details
-#     "stepllnoise" - RAND+StepLL method from https://arxiv.org/abs/1705.07204
-# ${ADV_EPS} - size of adversarial perturbation, ignored when method is none
-python eval_on_adversarial.py \
---model_name=${MODEL_NAME} \
---checkpoint_path=${CHECKPOINT_PATH} \
---dataset_dir=${DATASET_DIR} \
---batch_size=50 \
---adversarial_method=${ADV_METHOD} \
---adversarial_eps=${ADV_EPS}
-```
-
-Below is an example how to evaluate one of the models on RAND+StepLL adversarial
-examples:
-
-```bash
-# Download checkpoint
-CHECKPOINT_DIR=/tmp/checkpoints
-mkdir ${CHECKPOINT_DIR}
-wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
-tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
-mv ens_adv_inception_resnet_v2.ckpt* ${CHECKPOINT_DIR}
-rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
-
-# Run evaluation
-python eval_on_adversarial.py \
---model_name=inception_v3 \
---checkpoint_path=${CHECKPOINT_DIR}/ens_adv_inception_resnet_v2.ckpt \
---dataset_dir=${DATASET_DIR} \
---batch_size=50 \
---adversarial_method=stepllnoise \
---adversarial_eps=16
-```
-
-
-
-# Learning to Protect Communications with Adversarial Neural Cryptography
-
-This is a slightly-updated model used for the paper
-["Learning to Protect Communications with Adversarial Neural
-Cryptography"](https://arxiv.org/abs/1610.06918).
-
-> We ask whether neural networks can learn to use secret keys to protect
-> information from other neural networks. Specifically, we focus on ensuring
-> confidentiality properties in a multiagent system, and we specify those
-> properties in terms of an adversary. Thus, a system may consist of neural
-> networks named Alice and Bob, and we aim to limit what a third neural
-> network named Eve learns from eavesdropping on the communication between
-> Alice and Bob. We do not prescribe specific cryptographic algorithms to
-> these neural networks; instead, we train end-to-end, adversarially.
-> We demonstrate that the neural networks can learn how to perform forms of
-> encryption and decryption, and also how to apply these operations
-> selectively in order to meet confidentiality goals.



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
