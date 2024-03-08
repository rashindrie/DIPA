# Discriminative Sample-Guided and Parameter-Efficient Feature Space Adaptation for Cross-domain Few-Shot Learning

<p align="center">
  <img src="./figures/fsl.png" style="width:60%">
</p>


> [**Discriminative Sample-Guided and Parameter-Efficient Feature Space Adaptation for Cross-Domain Few-Shot Learning**](https://arxiv.org/abs/2403.04492),            
> Rashindrie Perera, Saman Halgamuge,        
> *CVPR 2024 ([arXiv 2403.04492](https://arxiv.org/abs/2403.04492))*  


## Pre-trained model checkpoints

We release following pre-trained checkpoints using Masked Image Modelling (MIM) for reproducibility.
1) Pre-trained on eight datasets: [MDL checkpoint](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=sTU2SepD5ePr4tfF8NHq11282382253&browser=true&filename=checkpoint_MDL.pth)
2) Pre-trained only on ImageNet-train set: [SDL checkpoint](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=u7f9hIqry6YHjBHF7FU611282382251&browser=true&filename=checkpoint_SDL.pth)

Additionally, the SDL-E checkpoints which were already available and used in our work can be accessed via below links:
1) MIM Pre-trained on ImageNet-full set: [SDL-E MIM checkpoint](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=aPEDmq1Xh6pxWq2uRDHZ11282382245&browser=true&filename=checkpoint_SDL_E_IBOT.pth).
2) DINO Pre-trained on ImageNet-full set: [SDL-E DINO checkpoint](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=UxsiEvaecI1Jl7C5oeXH11282382249&browser=true&filename=checkpoint_SDL_E_DINO.pth).

## Prerequisites

Please download and install [Pytorch 1.9.0](https://pytorch.org/) and [TensorFlow 2.6.0](https://tensorflow.org/). This code was tested on Python 3.8.6 and CUDA 11.1.1.

```
pip install -r requirements.txt
```

## Datasets

We utilize the [Meta-Dataset](https://github.com/google-research/meta-dataset) for our main results. Instructions for downloading and pre-processing Meta-Dataset can be found [here](https://github.com/google-research/meta-dataset#downloading-and-converting-datasets).
We provide a dataset class for Meta-Dataset to be used during pre-training under the `datasets` folder.
We also provide the label files created for MIM pre-training on Meta-Dataset here: [label_folder](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=J85UvGBFEesNxecpuXB811282382273&browser=true&filename=ibot_data.zip). 

## Pre-training

We mainly follow the hyperparameters provided for pre-training using [MIM](https://github.com/bytedance/ibot) while additionally following the author's recommendations to set teacher patch temperature to 0.04 instead of the default 0.07 provided in the source code. 
```

export NCCL_SOCKET_IFNAME="bond0.3027,p1p1.3027"
export NCCL_IB_HCA=mlx5_bond_0,mlx5_0
export NCCL_IB_GID_INDEX=7

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
export WORLD_SIZE=4

CHECKPOINT_DIR='./output_folder' # define path here

python -m torch.distributed.launch --use_env --nproc_per_node $WORLD_SIZE main_pretrain.py  \
  --dataset 'meta-dataset' \
  --data_path ${PATH_TO_META_DATASET_FOLDER} \
  --label_path ${PATH_TO_LABEL_FOLDER} \
  --output_dir ${CHECKPOINT_DIR} \
  --arch vit_small \
  --norm_last_layer false --use_fp16 True \
  --image_size 224 --local_crops_size 96 --patch_size 16 \
  --batch_size_per_gpu 128 \
  --epochs 800 \
  --shared_head true \
  --out_dim 8192 \
  --local_crops_number 10 \
  --global_crops_scale 0.25 1 \
  --local_crops_scale 0.05 0.25 \
  --pred_ratio 0 0.3 \
  --pred_ratio_var 0 0.2 \
  --teacher_temp 0.04 --teacher_patch_temp 0.04 --warmup_teacher_temp_epochs 30 --warmup_epochs 10 \
```

To pre-train an MDL backbone set `--dataset meta-dataset` else, to pre-train a SDL backbone set `--dataset imagenet`. Default choice is `meta-dataset`.

## Meta-Testing

### Meta-Dataset


Place/run the below code snippet which is required for using the MetaDataset readers, before running the evaluation scripts.
```
ulimit -n 50000

export META_DATASET_ROOT='/data/gpfs/projects/punim1193/few-shot-experiments/simple-cnaps/meta-dataset/'
export META_DATASET_ROOT='/data/gpfs/projects/punim1193/public_datasets/meta-dataset/'

export DATASRC='/data/gpfs/projects/punim1193/public_datasets/meta-dataset/data'
export SPLITS='/data/gpfs/projects/punim1193/public_datasets/meta-dataset/splits'
export RECORDS='/data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data'
```

### Evaluation:
```
CUDA_VISIBLE_DEVICES=0 python -u test_extractor.py \
    --pretrained_setting 'MDL' --test_type 'standard' \
    --out_dir ${RESULTS_PATH} --checkpoint_path ${PATH_TO_CHECKPOINT} 
```
Ensure that `checkpoint_path` points to the pre-trained checkpoint and `out_dir` points to the results folder in which you need to save evalution results.


To reproduce the `N-way-K-shot tasks` results presented in main text, set the test_type as `standard`:
```
--test_type 'standard'
```
To reproduce the  `varying-way-5-shot` results in main text, set test_type as `5shot`: 
```
--test_type '5shot'
```

For running evaluation on an MDL pre-trained checkpoint, set `pretrained_setting` as `MDL`. 
```
--pretrained_setting 'MDL' 
```

Otherwise, use `SDL` or `SDL_E` for running evaluation on other settings.
```
--pretrained_setting 'SDL' or --pretrained_setting 'SDL_E' 
```

## Additional analysis on CIFAR-FS and mini-ImageNet


## Datasets

We utilize CIFAR-FS, and Mini-ImageNet for additional evaluations. Please refer our Supplementary material for results from the additional evaluations.

- CIFAR-FS can be downloaded using the command:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Lq2USoQmbFgCFJlGx3huFSfjqwtxMCL8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Lq2USoQmbFgCFJlGx3huFSfjqwtxMCL8" -O cifar_fs.tar && rm -rf /tmp/cookies.txt

tar -xvf cifar_fs.tar cifar_fs/
```

- Mini-ImageNet can be downloaded [here](https://cseweb.ucsd.edu/~weijian/static/datasets/mini-ImageNet/).
```
tar -xvf MiniImagenet.tar.gz miniimagenet_224/
```

We also provide dataset classes for above datasets under the `datasets` folder.


## Meta-Testing
```
CUDA_VISIBLE_DEVICES=0 python -u test_extractor_others.py \
    --n_way ${N_WAY} --k_shot ${K_SHOT} --dataset ${DATASET_NAME} \
    --checkpoint_path ${PATH_TO_CHECKPOINT} --data_path ${DATASET_PATH} --out_dir ${RESULTS_PATH} 
```
Set `${DATASET_NAME}` as `cifar-fs` or as `mini_imaget` for CIFAR-FS and mini-ImageNet datasets, respectively. 
Here, the convention is to evaluate `5-way-5-shot` or `5-way-1-shot`. Define `N_WAY`, and `K_SHOT` according to the specific task you need to evaluate.

For example, for evaluating the 5-way-5-shot setting for cifar-fs dataset:

```
CUDA_VISIBLE_DEVICES=0 python -u test_extractor_others.py \
    --n_way 5 --k_shot 5 --dataset cifar_fs \
    --checkpoint_path ${PATH_TO_CHECKPOINT} --data_path ${DATASET_PATH} --out_dir ${RESULTS_PATH} 
```

## Citation

If you find our project helpful, please consider to cite our paper:

```
@misc{perera2024discriminative,
      title={Discriminative Sample-Guided and Parameter-Efficient Feature Space Adaptation for Cross-Domain Few-Shot Learning}, 
      author={Rashindrie Perera and Saman Halgamuge},
      year={2024},
      eprint={2403.04492},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
