# TESA: Task-Agnostic Embedding of Scene Graphs using Vision Alignment
PyTorch implementation of the paper.

## Setup

Install Conda. The environment `cuda` is used.

Exception: voltron preprocessing (further info below).

### Environment 'cuda'
Install the environment using:
> conda env create -f condaenv_cuda.yaml

> pip install git+https://github.com/openai/CLIP.git
### Environment for voltron preprocessing
In case you want to use the voltron-based models, preprocessing has to be done in python 3.8.
 
Create a python 3.8 environment and run:
> pip install "cython<3.0.0" wheel

> pip install voltron-robotics

## Data Preparation
- copy the folder `./data` into a location with enough disk space
- specify its location in .env (see .env.template)
- the full structure of the `./data` folder is as follows: 
```
processed/ (filled during preproccessing)
raw/
  coco/
    train2017/
    val2017/
  gqa/
  psg/
  vg/
  vqa/
unified_json/ (files as in ./data)
psg_captions/ (files as in ./data)
```
- Download and insert datasets into the respective folders. You might not need all downloads based on which version of TESA you want to train.
  - [Common Objects in COntext (COCO)](https://cocodataset.org/#download),
  - [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)
  - [PSG](https://psgdataset.org/)
  - VG150 (Images: [Part 1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [Part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip); [Annotations](https://drive.google.com/file/d/1aGwEu392DiECGdvwaYr-LgqGLmWhn8yD/view?usp=sharing) (Links copied from [SPAN](https://github.com/yrcong/Learning_Similarity_between_Graphs_Images)))
  - [VQA](https://visualqa.org/)

- preproccessing: run `preprocessor.py` with all combinations of datasets and vision models that you want to use for training.
This will store the image embeddings for a faster training process.


## Training and Evaluation
Adjust the `config.yaml`,
Run main.py

For only doing evaluation, use `--eval`. In that case, the old config (saved from training) is loaded and overwritten by `config_eval.yaml`.
