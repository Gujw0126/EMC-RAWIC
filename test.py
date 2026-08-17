import os
import torch
import torch.nn as nn
import rawpy
import numpy as np
import math
import torch.nn.functional as F
import time
from train import Loss
import collections
from models import RawCompressor_color_scale_bit
import random
from typing import List
torch.set_num_threads(1)

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, out_compress, shape):
        out = {}
        yz_likelihoods = out_compress['yz_likelihoods']
        x_likelihoods = out_compress['x_likelihoods']
        B, C, H, W = shape
        num_pixels = B * C * H * W
        
        out["yz_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in yz_likelihoods.values()
        )
        
        out["x_bpp"] = x_likelihoods.sum() / num_pixels
        out["loss"] = out["x_bpp"] + out["yz_bpp"]
        return out
    


def split_images(image_arr, patch_size, black:int):
    img_list = []
    H, W = image_arr.shape
    h_num = math.ceil(H / patch_size)
    w_num = math.ceil(W / patch_size)
    for i in range (0, h_num):
        for j in range(0, w_num):
            h_limit = min((i+1)*patch_size, H)
            w_limit = min((j+1)*patch_size, W)
            img_patch = image_arr[i*patch_size:h_limit, j*patch_size:w_limit]
            img_list.append(img_patch)
    return img_list


def merge_patches(decoded_patches, img_h, img_w, patch_size):
    device = decoded_patches[0].device
    img = torch.zeros([img_h, img_w], device=device)
    h_num = math.ceil(img_h / patch_size)
    w_num = math.ceil(img_w / patch_size)
    for i in range(h_num):
        for j in range(w_num):
            h_limit = min((i+1)*patch_size, img_h)
            w_limit = min((j+1)*patch_size, img_w)
            img[i*patch_size:h_limit, j*patch_size:w_limit] = decoded_patches[i*w_num+j]
    img = img.to('cpu')
    return img
    

def test_inference(img_arr, model, black_level, white_level, criterion):
    #print(img_arr.shape)
    img_list = split_images(img_arr, 512, black_level)
    bpp_avg = 0
    y_bpp_avg = 0
    z_bpp_avg = 0
    count = len(img_list)
    avg_enc_time = 0
    avg_dec_time = 0
    with torch.no_grad():
        for i in range(count):
            img_patch = img_list[i].to('cuda').unsqueeze(0).unsqueeze(0) 
            black_level = black_level.to('cuda')
            white_level = white_level.to('cuda')

            black_level = black_level.reshape(1)
            white_level = white_level.reshape(1)
            B, C, H, W = img_patch.shape
            out_forward = model.inference(img_patch, black=black_level, white=white_level)
            out_criterion = criterion(out_forward, img_arr.unsqueeze(0).unsqueeze(0).shape)

            bpp_avg += out_criterion['loss']
            bpp = out_criterion['loss']
            y_bpp_avg += out_criterion['yz_bpp']


    print(f'bpp_avg:{bpp_avg:.4f}')
    print(f'yz_bpp_avg:{y_bpp_avg:.4f}')
    return bpp_avg



def test_compress(img_arr, model, black_level, white_level, split=True):
    enc_start = time.time()
    img_list = split_images(img_arr, 512, black_level)
    bpp_avg = 0
    bpp_y_avg = 0
    bpp_z_avg = 0
    count = len(img_list)
    img_h, img_w = img_arr.shape
    #return a list contains all out_compress 
    out_compress_list = []
    
    with torch.no_grad():
        for i in range(count):
            #compress patch
            img_patch = img_list[i].to('cuda').unsqueeze(0).unsqueeze(0)
            black_level = black_level.to('cuda')
            white_level = white_level.to('cuda')
            black_level = black_level.reshape(1)
            white_level = white_level.reshape(1)
            B, C, H, W = img_patch.shape
            out_compress = model.compress(img_patch, black_level, white_level)  
            out_compress_list.append(out_compress)
              
            x_bpp = 0 
            
            #calculate bpp, x part
            x_strings = out_compress['strings'][1]
            for color_strings in x_strings:
                if isinstance(color_strings, list):
                    for part_strings in color_strings:
                        if isinstance(part_strings, list):
                            for bit_strings in part_strings:
                                x_bpp += len(bit_strings)*8/img_h/img_w
                        else:
                            x_bpp += len(part_strings)*8/img_h/img_w                               
                else:
                    x_bpp += len(color_strings)*8/img_h/img_w  

            y_bpp = len(out_compress['strings'][0][0][0])*8/img_h/img_w
            z_bpp = len(out_compress['strings'][0][1][0])*8/img_h/img_w
            bpp_avg += x_bpp + y_bpp + z_bpp
            bpp_y_avg += y_bpp
            bpp_z_avg += z_bpp
    
    enc_time = time.time() - enc_start
    print(f'compress_bpp_avg: {bpp_avg:.4f}')
    print(f'compress_bpp_y_avg: {bpp_y_avg:.4f}')
    print(f'compress_bpp_z_avg: {bpp_z_avg:.4f}')
    print(f'enc_time: {enc_time:.4f}') 
    return bpp_avg, enc_time, out_compress_list, img_h, img_w           
           

def test_decompress(model, black_level, white_level, out_compress_list, img_h, img_w):
    dec_start = time.time()
    count = len(out_compress_list)
    decoded_patches = []
    with torch.no_grad():  
        for i in range(count): 
            black_level = black_level.to('cuda')
            white_level = white_level.to('cuda')
            black_level = black_level.reshape(1)
            white_level = white_level.reshape(1)
            out_decompress = model.decompress(shapes=out_compress_list[i]['shapes'], black=black_level, white=white_level, strings=out_compress_list[i]['strings'])
            decoded_patches.append(out_decompress)
    decoded_x = merge_patches(decoded_patches, img_h, img_w, 512)
    dec_time = time.time() - dec_start
    print(f'dec_time:{dec_time:.4f}')
    return decoded_x, dec_time
 
    
def main(root, checkpoint, split=False):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)


    model = RawCompressor_color_scale_bit(bit=14,scale=2)
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    model.update()
    model.to("cuda")
    model.eval()
    avg_bpp_t = 0
    avg_bpp_r = 0
    avg_enc_time = 0
    avg_dec_time = 0
    count = 0
    criterion = Loss()
    for i, filename in enumerate(os.listdir(root)):
        filepath = os.path.join(root, filename)
        print(f'i={i}, filename:{filename}')
        with rawpy.imread(filepath) as raw:
            raw_data = raw.raw_image.astype(np.int16) 
            raw_data = torch.from_numpy(raw_data).to(torch.int16) 
            #for 14 bit, if raw image is 12 bit, max=4095
            raw_data = torch.clamp(raw_data, min=0, max=16383)
            print(f'raw_min:{raw_data.min()}')
            print(f'raw_max:{raw_data.max()}')
            #for 14 bit, if raw image is 12 bit, white=4095
            black_level = torch.tensor(0)
            white_level = torch.tensor(16383)
            avg_bpp_t += test_inference(raw_data, model, black_level, white_level, criterion)
       
            bpp_avg, enc_time, out_compress_list, img_h, img_w = test_compress(raw_data, model, black_level, white_level)
            
            decoded_x, dec_time = test_decompress(model, black_level, white_level, out_compress_list, img_h, img_w)
            print((decoded_x-raw_data).sum())
            count += 1
            avg_enc_time += enc_time
            avg_dec_time += dec_time
            avg_bpp_r += bpp_avg
            print(f'current_avg_t:{avg_bpp_t/count:.4f}')
            print(f'current_avg_r:{avg_bpp_r/count:.4f}')
    print(f'avg_bpp_r:{avg_bpp_r/count:.4f}')    
    print(f'avg_bpp_t:{avg_bpp_t/count:.4f}')
    print(f'avg_enc_time:{avg_enc_time/count:.4f}')
    print(f'avg_dec_time:{avg_dec_time/count:.4f}')

    
if __name__ == "__main__":
    main("path to testset","path to checkpoint")
    