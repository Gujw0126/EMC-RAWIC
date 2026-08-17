import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F 
from typing import List
_LOG_SCALES_MIN = -7

#modified from https://github.com/fab-jul/L3C-PyTorch/blob/master/src/modules/prob_clf.py
def log_softmax(logit_weight, dim):
    m,_ = torch.max(logit_weight, dim=dim, keepdim=True)
    return logit_weight - m - torch.log(torch.sum(torch.exp(logit_weight - m),dim=dim, keepdim=True))


def log_sum_exp(log_weight, dim):
    m, _ = torch.max(log_weight, dim=dim)
    m_keep, _ = torch.max(log_weight, dim=dim, keepdim=True)
    return log_weight.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)


class LogisticMixture(nn.Module):
    def __init__(self):
        super().__init__()
    
    def param_prepare(self, x:Tensor, params:Tensor, param_num:int, ch_num:int):
        if param_num==3:
            B, C, H, W = x.shape
            C_param = params.shape[1]
            K = C_param//C//param_num            
            x_out = x.reshape([B,C,1,H,W])
            miu, log_scale, weight = torch.chunk(params, chunks=3, dim=1)
            #miu:[B,C*K,H,W]
            log_scale = torch.clamp(log_scale, min=_LOG_SCALES_MIN)
            miu = miu.reshape([B,C,K,H,W])
            log_scale = log_scale.reshape([B,C,K,H,W])
            weight = weight.reshape([B,C,K,H,W])
        elif param_num==4 and ch_num==4:
            B, C, H, W =  x.shape
            x_out = x.reshape([B,C,1,H,W])
            assert C==4
            C_param = params.shape[1]
            K = C_param//18
            miu, log_scale, weight = torch.chunk(params[:,:12*K,:,:], 3, 1)
            lambda1, lambda2, lambda3, lambda4, lambda5, lambda6 = torch.chunk(params[:,12*K:,:,:], 6, dim=1)
            #lamda_a [B,K,H,W]
            miu_r = miu[:,0:K, :, :]
            miu_b = miu[:,K:2*K:,:,:]
            miu_g1 = miu[:,2*K:3*K,:,:]
            miu_g2 = miu[:,3*K:4*K,:,:]
            
            miu_b = miu_b + torch.sigmoid(lambda1) * x[:,0:1,:,:]
            miu_g1 = miu_g1 + torch.sigmoid(lambda2) * x[:,0:1,:,:] + torch.sigmoid(lambda3) * x[:,1:2,:,:]
            miu_g2 = miu_g2 + torch.sigmoid(lambda4) * x[:,0:1,:,:] + torch.sigmoid(lambda5) * x[:,1:2,:,:] + torch.sigmoid(lambda6) * x[:,2:3,:,:]
            miu = torch.concatenate([miu_r, miu_b, miu_g1, miu_g2], dim=1)
            log_scale = torch.clamp(log_scale, min=_LOG_SCALES_MIN)
            miu = miu.reshape([B,C,K,H,W])
            log_scale = log_scale.reshape([B,C,K,H,W])
            weight = weight.reshape([B,C,K,H,W])
        elif param_num==4 and ch_num==3:
            B, C, H, W = x.shape
            x_out = x.reshape([B, C, 1, H, W])
            assert C==3
            C_param = params.shape[1]
            K = C_param // 18
            miu, log_scale, weight = torch.chunk(params[:,:9*K,:,:], 3, 1)
            lambda1, lambda2, lambda3 = torch.chunk(params[:,9*K:12*K,:,:], 3, 1)
            miu_r = miu[:,0::K,:,:]
            miu_b = miu[:,K:2*K,:,:]
            miu_g = miu[:,2*K:3*K,:,:]
            miu_b = miu_b + torch.sigmoid(lambda1) * x[:,0:1,:,:]
            mig_g = miu_g + torch.sigmoid(lambda2) * x[:,0:1,:,:] + torch.sigmoid(lambda3) * x[:,1:2,:,:]
            miu = torch.concatenate([miu_r, miu_b, miu_g], dim=1)
            miu = miu.reshape([B, C, K, H, W])
            log_scale = log_scale.reshape([B, C, K, H, W])
            weight = weight.reshape([B, C, K, H, W])
        else:
            raise RuntimeError("param_num must be 3 or 4")
        return x_out, miu, log_scale, weight


    def forward(self, quantized_x:Tensor, params:Tensor, param_num:int, ch_num:int, bin, x_min:float, x_max:float):
        assert not torch.isnan(params).any()
        x_out, miu, log_scale, weight = self.param_prepare(quantized_x, params, param_num, ch_num)
        scale = torch.exp(-log_scale)
        plus_in = scale*(x_out + bin/2 - miu)
        min_in = scale*(x_out - bin/2 - miu)
        cdf_plus = torch.sigmoid(plus_in)
        cdf_min = torch.sigmoid(min_in)
        cdf_delta = cdf_plus - cdf_min
        log_cdf_plus = plus_in - F.softplus(plus_in)  #for x_out==x_min; plus_in-F.softmax(plus_in)==log(sigmoid(plus_in))
        log_one_minus_cdf_min = -F.softplus(min_in)  #for x_out==x_max; -F.softmax(min_in)==log(1-sigmoid(min_in))

        x_min = x_min + 1e-9
        x_max = x_max - 1e-9
        #NOTE: three conditions
        #if x < x_min:    cond_C
        #        log_cdf_plus      out_c
        #elif x > x_max:  cond_B
        #        log_one_minus_cdf_min  out_B
        #else
        #        log(cdf_delta)        out_A
        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
        cond_B = (x_out > x_max).float()
        out_B = (cond_B*log_one_minus_cdf_min + (1. - cond_B)*out_A)
        cond_C = (x_out < x_min).float()
        log_probs = cond_C * log_cdf_plus + (1. - cond_C) * out_B
        #log_probs = torch.clamp(log_probs,min=-25, max=25)
        #conbine with pi
        log_probs_weighted = log_probs.add(
            log_softmax(weight, dim=2)
        )

        neg_log_likelihood = -log_sum_exp(log_probs_weighted, dim=2)/math.log(2)
        #neg_log_likelihood = torch.clip(neg_log_likelihood, min=1e-3)
        return neg_log_likelihood
    