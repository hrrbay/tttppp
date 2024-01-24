# Turbo Table Tennis Point Probability Prediction

The goal of our project is to predict which player scores the next point in a table tennis match.
To this extent, we make use of the [data](https://lab.osai.ai/) and 
[network](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w53/Voeikov_TTNet_Real-Time_Temporal_and_Spatial_Video_Analysis_of_Table_Tennis_CVPRW_2020_paper.pdf) 
published by Voeikov et al at CVPR in 2020.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necessary libraries.

```bash
pip install -r requirements.txt
```

TODO dataset

## Usage

### Training Baseline 3D CNN
TODO
```bash
bash ./src/scripts/r3d_script.sh
```

### Training R3D Model
TODO
```bash
bash ./src/scripts/r3d_script.sh
```


### Training a linear layer on top of Hiera
In our [Hiera](https://github.com/facebookresearch/hiera) approach, we finetune a linear layer on top of the last intermediate outputs of the 51 million parameter Hiera Base model. 
We use a similar head to the original model, only changing the number of classes in the output for binary classification. We use a Hiera model checkpoint from pretraining and finetuning on the
Kinetics 400 dataset. The input to this model is and resized to dimension 16x224x224.

```bash
bash ./src/scripts/train_hiera.sh
```

### Training Table Tennis Transformer
We additionally trained a small Encoder-only transformer on tabular input, consisting of pose estimations and ball positions.
We reshape the input in the forward pass from (window_size, n_poses + ball position, 2) to (n_poses + ball position, window_size*2), 
embed the tokens to a hidden dimension of 128 and add a learned positional encoding. Our architecture consists of 2 layers and 4 attention heads.
After the encoder, we use the same head as in the Hiera finetuning approach.

```bash
bash ./src/scripts/train_tttransformer.sh
```

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
