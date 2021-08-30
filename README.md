# This Repository implement different General Adversarial Networks which can be used to create new images/characters

### Implement

#### 1. Self Attention-General Adversarial Network

https://arxiv.org/pdf/1805.08318.pdf

Implemented with Attention, Conv2DTranspose, hinge loss and spectral norm.

The SAGAN was trained in batchsize=64 and cost only 3GB GPU memory. It needs about 50000 steps for training.

#### 2. BIG-General Adversarial Network

https://arxiv.org/abs/1809.11096?context=cs.LG

The BIGGAN was trained in batchsize=64 and cost 16GB GPU memory (batchsize=32 cost 10GB GPU memory for 1080Ti). It needs only 10000 steps for training.


### Open sourced dataset
Dataset used to generate new Anime Characters.

It's an open source dataset and can be downloaded by below link.

url:https://www.kaggle.com/splcher/animefacedataset  


### Steps to Reproduce

#### 1. Download the dataset from the above URL.
#### 2. Clone this Repository
#### 3. Enter directory.
#### 3. pip3 install -r requirements.txt
#### 4. Run command python3 train.py --input-dir dir_path




 



