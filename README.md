# Turbo Table Tennis Point Probability Prediction (T3P3)

This repository provides code necessary to train T3P3 (IVU project WS23/24). 

Authors: Paul Plantosar, Jakob Sattler, Christian Thurner
## Installation
### Environment
Python 3.11.7 was used for all experiments and development. 
Install all packages given in `requirements.txt` and activate the environment **before** performing any other steps. 

### Getting the data
By default, `./t3p3_data` (with `.` being the **root-path** of this project) will be used as the default data-path for all scripts manipulating or reading data. Setting the environment-variable `T3P3_DATA` will replace the default path. 

Note that the data takes up about 220GB after extraction.

Run
```
./scripts/download_data.sh
```
to download and extract all data to the path as described above. This will directly provide the masks adapted from TTNet-output which are usable for T3P3.


## Training

In order to skip the next steps about training different models separately, simply run
```bash
./scripts/train_all.sh
```
to train all models. All scripts (including the following) will only train using the default seed (0).

### $\mu\ell$-net
You can train our baseline $\mu\ell$-net with
```bash
./scripts/train_muell.sh
```
The model uses input dimensions of 1x1x120x320 (CDHW).

### R3D 
We use [R3D_18](https://pytorch.org/vision/0.12/generated/torchvision.models.video.r3d_18.html) without pretraining and replace the head to perform binary classification. The model uses input dimensions of 3x16x112x112 (CDHW)
```bash
./scripts/train_r3d.sh
```


#### Training a linear layer on top of Hiera
In our [Hiera](https://github.com/facebookresearch/hiera) approach, we finetune a linear layer on top of the last intermediate outputs of the 51 million parameter Hiera Base model. 
We use a similar head to the original model, only changing the number of classes in the output for binary classification. We use a Hiera model checkpoint from pretraining and finetuning on the
Kinetics 400 dataset. The model uses input dimensions of 3x16x224x224 (CDHW).

```bash
./scripts/train_hiera.sh
```

#### Training Table Tennis Transformer
We additionally trained a small Encoder-only transformer on tabular input, consisting of pose estimations and ball positions.
We reshape the input in the forward pass from (window_size, n_poses + ball position, 2) to (n_poses + ball position, window_size*2), 
embed the tokens to a hidden dimension of 128 and add a learned positional encoding. Our architecture consists of 2 layers and 4 attention heads.
After the encoder, we use the same head as in the Hiera finetuning approach.

```bash
./scripts/train_tttransformer.sh
```

### Evaluating trained models
To evaluate trained models you first have to download them [here](TODO!!!) and place all `.pth`-files into `./models`.

You can then evaluate all of the networks in the same way as training them, but replace `train` with `eval` in the script name. So, e.g., to evaluate on all networks run 
```bash
./scripts/eval_all.sh
```
Each script uses a checkpoint provided in `./models`.