# Code for EMC_RAWIC

## 1. environment
conda create -n emc_rawic -f environment.yml   
conda activate emc_rawic

install torchac(CUDA):https://github.com/fab-jul/L3C-PyTorch/
install torchac to ./models/torchac

## 2. train model
train:
CUDA_VISIBLE_DEVICES=xxx python train.py -d path_to_RAW --cuda -e 300 --save --save_path \ path_to_savepath --batch-size 6  --[nikon|sony|samsung|olympus|] >> log_file.log

finetune:
CUDA_VISIBLE_DEVICES=xxx python train.py -d path_to_RAW --cuda -e 100 --save --save_path \ path_to_savepath --batch-size 6  --[nikon|sony|samsung|olympus|] \
--checkpoint path_to_checkpoint -lr 1e-5 >> log_file.log

## 3. test model
CUDA_VISIBLE_DEVICES=xxx python test.py
