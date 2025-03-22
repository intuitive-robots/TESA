# TESA: Task-Agnostic Embedding of Scene Graphs using Vision Alignment
PyTorch implementation of the paper.

## Setup
two conda environments are used: "cuda" and "p38". p38 is only used for preprocessing of voltron.

Install Conda. 
### Environment 'cuda'
Install the environment using:
> conda env create -f condaenv_cuda.yaml

> pip install git+https://github.com/openai/CLIP.git
### Environment 'p38' (only for voltron preprocessing)
p38 uses python 3.8 and 
pip install "cython<3.0.0" wheel
pip install voltron-robotics

## Data Preparation
Copy .env.template into .env - specify correct dataset folder.
Download the datasets and place them in the following structure of data folder:


processed/
raw/
  coco/
    train2017/ (download from https://cocodataset.org/#download)
    val2017/   (download from https://cocodataset.org/#download)
  gqa/ (insert gqa dataset)
  psg/ (insert psg dataset)
  vg/  (insert vg dataset)
  vqa/ (insert vqa dataset)
unified_json/ (files already included here)
psg_captions/ (files already included here)

- run `preprocessor.py` with all combinations of datasets and vision models that you want to use for training.
This will store the image embeddings for a faster training process.


## Training and Evaluation
Adjust the `config.yaml`,
Run main.py

For only doing evaluation, use `--eval`. In that case, the old config (saved from training) is loaded and overwritten by `config_eval.yaml`.
