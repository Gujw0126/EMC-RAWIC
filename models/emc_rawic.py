import torch
import torch.nn as nn
import torch.nn.functional as F
from .torchac import torchac
from compressai.layers import conv1x1
import math
from .modules import LatentCompressor, conv1x1, ResidualBlock, ResBlock_1x1
from .entropy import LogisticMixture


class Extractor(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.trans = nn.Sequential(
        conv1x1(input_dim, dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        )
       
    def forward(self, x):
        return self.trans(x)


#content-adaptive extractor
class SpatialExtractor(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.trans = nn.Sequential(
        conv1x1(input_dim, dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        )
        
        self.attn = nn.Sequential(
        conv1x1(input_dim, dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        ResidualBlock(in_ch=dim, out_ch=dim),
        conv1x1(in_ch=dim, out_ch=dim),
        nn.Sigmoid()
        )
        
        self.in_conv =  conv1x1(in_ch=input_dim, out_ch=dim)
       
    
    def forward(self, x):
        return self.trans(x)*self.attn(x) + self.in_conv(x)

    
class ParamAgg(nn.Module):
    def __init__(self, num_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ResBlock = ResBlock_1x1(num_ch)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, out_ch),
        )
         
    def forward(self, input):
        out = self.ResBlock(input)
        out = self.layers(out)
        return out


#padding used for inference
def pad_x(x):
    B,C,h_valid, w_valid = x.shape
    pad_down_value = torch.mean(x[:,:,h_valid-1:,:], keepdim=True, dim=-1)
    pad_right_value = torch.mean(x[:,:,:,w_valid-1:], keepdim=True, dim=2)
    h_pad = math.ceil(h_valid / 64)*64 - h_valid
    w_pad = math.ceil(w_valid / 64)*64 - w_valid
    pad_down = pad_down_value.repeat(1,1,h_pad, w_valid)
    pad_right = pad_right_value.repeat(1,1,h_valid, w_pad)
    pad_other = ((pad_down_value + pad_right_value)/2).repeat(1,1,h_pad,w_pad)
    x_pad = torch.zeros([B,C,h_valid+h_pad, w_valid+w_pad], device=x.device)
    x_pad[:,:,0:h_valid,0:w_valid] = x
    x_pad[:,:,h_valid:h_valid+h_pad,0:w_valid] = pad_down
    x_pad[:,:,0:h_valid,w_valid:w_valid+w_pad] = pad_right
    x_pad[:,:,h_valid:h_valid+h_pad,w_valid:w_valid+w_pad] = pad_other
    
    x = F.pad(x, pad=[0,w_valid%2,0,h_valid%2],mode='constant',value=0) 
    return x, x_pad


#full model of EMC-RAWIC
class RawCompressor_color_scale_bit(nn.Module):
    def __init__(self, latent_dim=192, context_dim=256, ch_dim=128, spatial_dim=64, bit_dim=64, K=5, scale=2, bit=14):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = scale
        self.logistic_mix = LogisticMixture()
        self.spatial_trans = nn.ModuleList(
            [nn.ModuleList(
                [
                    nn.ModuleList([
                        SpatialExtractor(input_dim=p, dim=spatial_dim)
                    for p in range(1,4)])
                for s in range(scale)] 
            )
             for color in range(4)]
        )
        
        #parameter network for base sub-image, MSP
        self.aggs_first_most = nn.ModuleList([ParamAgg(num_ch=context_dim+(ch_dim if color>0 else 0), out_ch=3*K) for color in range(4)])
        
        #parameter network for other MSP
        self.aggs_most = nn.ModuleList(
            [nn.ModuleList(
                [
                    nn.ModuleList([
                        ParamAgg(num_ch=context_dim+(ch_dim if color>0 else 0)+spatial_dim, out_ch=3*K)
                    for p in range(1,4)])
                for s in range(scale)] 
            )
             for color in range(4)]
        )

        #parameter network for base sub-image, LSP
        self.aggs_first_least = nn.ModuleList([ParamAgg(num_ch=context_dim+ bit_dim +(ch_dim if color>0 else 0), out_ch=3*K) for color in range(4)])
        #parameter network for base sub-image, LSP
        self.aggs_least = nn.ModuleList(
            [nn.ModuleList(
                [
                    nn.ModuleList([
                        ParamAgg(num_ch=context_dim + bit_dim +(ch_dim if color>0 else 0) + spatial_dim, out_ch=3*K)
                    for p in range(1,4)])
                for s in range(scale)] 
            )
             for color in range(4)]
        )

        #global context
        self.latent_compress = LatentCompressor(in_ch=4, num_ch=latent_dim,context_dim=context_dim)
        
        self.color_trans = nn.ModuleList([Extractor(input_dim=i, dim=ch_dim) for i in range(1,4)])
        self.bit_trans = nn.ModuleList([
            nn.ModuleList([
                Extractor(input_dim=1, dim=bit_dim)
            for s in range(scale)])
         for color in range(4)])
        self.bit=bit

    #designed for RGGB CFA
    def pack_raw(self, raw):
        B, C, H, W = raw.shape
        rbgg = torch.zeros([B, 4*C, H//2, W//2], device=raw.device)
        #red
        rbgg[:,0,:,:] = raw[:, 0, 0::2, 0::2]
        #blue
        rbgg[:,1,:,:] = raw[:, 0, 1::2, 1::2]
        #green
        rbgg[:,2,:,:] = raw[:, 0, 0::2, 1::2]
        rbgg[:,3,:,:] = raw[:, 0, 1::2, 0::2]
        return rbgg

    def unpack_raw(self, rbgg):
        B, C, H, W = rbgg.shape
        raw = torch.zeros([B, C//4, 2*H, 2*W], device=rbgg.device)
        raw[:,0,0::2, 0::2] = rbgg[:,0,:,:]
        raw[:,0,1::2, 1::2] = rbgg[:,1,:,:]
        raw[:,0,0::2, 1::2] = rbgg[:,2,:,:]
        raw[:,0,1::2, 0::2] = rbgg[:,3,:,:]
        return raw
    

    #merge between scales
    def merge_scale(self, x):
        B, C, H, W = x.shape
        y = torch.zeros([B, C//4, 2*H, 2*W], device=x.device)
        c = C//4
        y[:,:,0::2,0::2] = x[:,0:c,:,:]
        y[:,:,0::2,1::2] = x[:,c:2*c,:,:]
        y[:,:,1::2,0::2] = x[:,2*c:3*c,:,:]
        y[:,:,1::2,1::2] = x[:,3*c:4*c,:,:]
        return y


    def forward(self, x, black, white):
        x = self.pack_raw((x-black[:,None,None,None])/(white[:,None,None,None]-black[:,None,None,None]))  #rbgg
        total_scale = self.scale
        out_latent = self.latent_compress(x)
        most_likelihoods = torch.zeros_like(x)
        least_likelihoods = torch.zeros_like(x)
        prior = out_latent['prior']
        split_scale = 2**(self.bit//2)

        for i in range(4):
            if i == 0:
                color_context = prior
            else:
                channel_context = self.color_trans[i-1](x[:,:i,:,:])
                color_context = torch.concat([prior, channel_context], dim=1)
            
            #base part
            target = x[:, i:i+1, 0::(2**total_scale), 0::(2**total_scale)]
            support_context = color_context[:, :, 0::(2**total_scale), 0::(2**total_scale)]  
            
            #MSP likelihoods
            target_most = torch.round(target*(white[:,None,None,None]-black[:,None,None,None])) // split_scale / (split_scale-1)
            most_params = self.aggs_first_most[i](support_context)
            most_likelihoods[:,i:i+1, 0::(2**total_scale), 0::(2**total_scale)] = self.logistic_mix(target_most, most_params,  param_num=3, ch_num=3, bin=1./(split_scale-1), x_min=0, x_max=1)
            
            #LSP likelihoods
            target_least = torch.round(target*(white[:,None,None,None]-black[:,None,None,None])) % split_scale / (split_scale-1)
            bit_context = self.bit_trans[i][total_scale-1](target_most)
            least_params = self.aggs_first_least[i](torch.concat([support_context, bit_context], dim=1))
            least_likelihoods[:,i:i+1, 0::(2**total_scale), 0::(2**total_scale)] = self.logistic_mix(target_least, least_params,  param_num=3, ch_num=3, bin=1./(split_scale-1), x_min=0, x_max=1)
            
            # other sub-images
            space_support = target
            for scale in reversed(range(1, total_scale+1)):
                step = 2**scale
                B, _, _, _ = x.shape
                for space in range(1, 4):                    #3 parts in each scale
                    target = x[:, i:i+1, (space//2)*(step//2)::step, (space%2)*(step//2)::step]    
                    support_latent = color_context[:, :, (space//2)*(step//2)::step, (space%2)*(step//2)::step]
                    space_context = self.spatial_trans[i][scale-1][space-1](space_support)    
                    support_context = torch.concat([support_latent, space_context], dim=1)        #support context = support_latent + space context
                    # MSP likelihoods
                    target_most = torch.round(target*(white[:,None,None,None]-black[:,None,None,None])) // split_scale / (split_scale-1)
                    most_params = self.aggs_most[i][scale-1][space-1](support_context)
                    most_part_likelihoods = self.logistic_mix(target_most, most_params, param_num=3, ch_num=3, bin=1./(split_scale-1), x_min=0, x_max=1)
                    most_likelihoods[:, i:i+1, (space//2)*(step//2)::step, (space%2)*(step//2)::step] = most_part_likelihoods
                    
                    # LSP likelihoods
                    target_least = torch.round(target*(white[:,None,None,None]-black[:,None,None,None])) % split_scale / (split_scale-1)
                    bit_context = self.bit_trans[i][scale-1](target_most)
                    least_params = self.aggs_least[i][scale-1][space-1](torch.concat([support_context, bit_context], dim=1))
                    least_part_likelihoods = self.logistic_mix(target_least, least_params, param_num=3, ch_num=3, bin=1./(split_scale-1), x_min=0, x_max=1)
                    least_likelihoods[:, i:i+1, (space//2)*(step//2)::step, (space%2)*(step//2)::step] = least_part_likelihoods
                    
                    space_support = torch.concat([space_support, target], dim=1)   #update space_support
                
                space_support = self.merge_scale(space_support)  #merge between scales  

        x_likelihoods = torch.concat([most_likelihoods, least_likelihoods], dim=1)
        return {
            "x_likelihoods": x_likelihoods,
            "yz_likelihoods": out_latent["likelihoods"],
            "aux_loss":self.aux_loss(),
            "prior":prior
        }


    #inference patch
    def inference(self, x, black, white):
        x = self.pack_raw((x-black[:,None,None,None])/(white[:,None,None,None]-black[:,None,None,None]))  #rbgg
        _,_,h_valid, w_valid = x.shape
        x_ar_pad, x_vae_pad = pad_x(x)                          #padding,
        _,_, H, W = x_ar_pad.shape
        total_scale = self.scale
        out_latent = self.latent_compress(x_vae_pad)
        most_likelihoods = torch.zeros_like(x_ar_pad)
        least_likelihoods = torch.zeros_like(x_ar_pad)
        prior = out_latent['prior'][:,:,:H,:W]
        split_scale = 2**(self.bit//2)
        for i in range(4):
            if i == 0:
                color_context = prior
            else:
                channel_context = self.color_trans[i-1](x_ar_pad[:,:i,:,:])
                color_context = torch.concat([prior, channel_context], dim=1)
            
            #base
            target = x_ar_pad[:, i:i+1, 0::(2**total_scale), 0::(2**total_scale)]
            support_context = color_context[:, :, 0::(2**total_scale), 0::(2**total_scale)]  
            
            target_most = torch.round(target*(white[:,None,None,None]-black[:,None,None,None])) // split_scale / (split_scale-1)
            most_params = self.aggs_first_most[i](support_context)
            most_likelihoods[:,i:i+1, 0::(2**total_scale), 0::(2**total_scale)] = self.logistic_mix(target_most, most_params,  param_num=3, ch_num=3, bin=1./(split_scale-1), x_min=0, x_max=1)
            
            target_least = torch.round(target*(white[:,None,None,None]-black[:,None,None,None])) % split_scale / (split_scale-1)
            bit_context = self.bit_trans[i][total_scale-1](target_most)
            least_params = self.aggs_first_least[i](torch.concat([support_context, bit_context], dim=1))
            least_likelihoods[:,i:i+1, 0::(2**total_scale), 0::(2**total_scale)] = self.logistic_mix(target_least, least_params,  param_num=3, ch_num=3, bin=1./(split_scale-1), x_min=0, x_max=1)
            

            space_support = target
            for scale in reversed(range(1, total_scale+1)):
                step = 2**scale
                B, _, _, _ = x_ar_pad.shape
                for space in range(1, 4):                   
                    target = x_ar_pad[:, i:i+1, (space//2)*(step//2)::step, (space%2)*(step//2)::step]    
                    support_latent = color_context[:, :, (space//2)*(step//2)::step, (space%2)*(step//2)::step]
                    space_context = self.spatial_trans[i][scale-1][space-1](space_support)    
                    support_context = torch.concat([support_latent, space_context], dim=1)        #support context = support_latent + space context
                    
                    target_most = torch.round(target*(white[:,None,None,None]-black[:,None,None,None])) // split_scale / (split_scale-1)
                    most_params = self.aggs_most[i][scale-1][space-1](support_context)
                    most_part_likelihoods = self.logistic_mix(target_most, most_params, param_num=3, ch_num=3, bin=1./(split_scale-1), x_min=0, x_max=1)
                    most_likelihoods[:, i:i+1, (space//2)*(step//2)::step, (space%2)*(step//2)::step] = most_part_likelihoods
                    target_least = torch.round(target*(white[:,None,None,None]-black[:,None,None,None])) % split_scale / (split_scale-1)
                    bit_context = self.bit_trans[i][scale-1](target_most)
                    least_params = self.aggs_least[i][scale-1][space-1](torch.concat([support_context, bit_context], dim=1))
                    least_part_likelihoods = self.logistic_mix(target_least, least_params, param_num=3, ch_num=3, bin=1./(split_scale-1), x_min=0, x_max=1)
                    least_likelihoods[:, i:i+1, (space//2)*(step//2)::step, (space%2)*(step//2)::step] = least_part_likelihoods
                    
                    space_support = torch.concat([space_support, target], dim=1)   
                
                space_support = self.merge_scale(space_support)     
        x_likelihoods = torch.concat([most_likelihoods, least_likelihoods], dim=1)
        return {
            "x_likelihoods": x_likelihoods[:,:,:h_valid,:w_valid],
            "yz_likelihoods": out_latent["likelihoods"],
            "aux_loss":self.aux_loss(),
            "prior":prior
        }
    
    #single image compress
    def compress(self, x, black:int, white:int):
        x = self.pack_raw((x-black)/(white-black))  #rbgg
        _,_,h_valid, w_valid = x.shape
        x_ar_pad, x_vae_pad = pad_x(x)                      
        _,_, H, W = x_ar_pad.shape
        total_scale = self.scale
        out_latent_compress = self.latent_compress.compress(x_vae_pad)
        prior = out_latent_compress['prior'][:,:,:H,:W]
        split_scale = 2**(self.bit//2)
        x_strings = []
        bin = 1/(split_scale-1)
        sym_value = torch.linspace(bin / 2, 1 - bin / 2, split_scale-1, dtype=torch.float32, device=x.device)  
        value_min = torch.tensor([-1], device=x.device)
        value_max = torch.tensor([2], device=x.device)
        sym_value = torch.concatenate([value_min, sym_value, value_max], dim=0)
        
        for i in range(4):
            if i == 0:
                color_context = prior
            else:
                channel_context = self.color_trans[i-1](x_ar_pad[:,:i,:,:])
                color_context = torch.concat([prior, channel_context], dim=1)
            
            space_strings = []
        
            target = x_ar_pad[:, i:i+1, 0::(2**total_scale), 0::(2**total_scale)]
            support_context = color_context[:, :, 0::(2**total_scale), 0::(2**total_scale)]  
            
            target_most = torch.round(target*(white-black)) // split_scale / (split_scale-1)
            most_params = self.aggs_first_most[i](support_context)
            means, scales, weights = most_params.chunk(3,1)
            sym = torch.round(target_most*(split_scale-1)).reshape(-1).to(torch.int16).to('cpu')  

            weights = torch.softmax(weights, dim=1)
            means = means.contiguous()
            scales = scales.contiguous()
            weights = weights.contiguous()
            space_strings.append(torchac.encode_logistic_mixture(sym_value, means, scales, weights, sym))
            
            
            target_least = torch.round(target*(white-black)) % split_scale / (split_scale-1)
            bit_context = self.bit_trans[i][total_scale-1](target_most)
            least_params = self.aggs_first_least[i](torch.concat([support_context, bit_context], dim=1))
            means, scales, weights = least_params.chunk(3,1)
            
            sym = torch.round(target_least*(split_scale-1)).reshape(-1).to(torch.int16).to('cpu')  

            weights = torch.softmax(weights, dim=1)
            means = means.contiguous()
            scales = scales.contiguous()
            weights = weights.contiguous()
            space_strings.append(torchac.encode_logistic_mixture(sym_value, means, scales, weights, sym))

            space_support = target
            for scale in reversed(range(1, total_scale+1)):
                step = 2**scale
                B, _, _, _ = x_ar_pad.shape
                for space in range(1, 4):              
                    target = x_ar_pad[:, i:i+1, (space//2)*(step//2)::step, (space%2)*(step//2)::step]   
                    support_latent = color_context[:, :, (space//2)*(step//2)::step, (space%2)*(step//2)::step]
                    space_context = self.spatial_trans[i][scale-1][space-1](space_support)    
                    support_context = torch.concat([support_latent, space_context], dim=1)       
                    
                    target_most = torch.round(target*(white-black)) // split_scale / (split_scale-1)
                    most_params = self.aggs_most[i][scale-1][space-1](support_context)
                    means, scales, weights = most_params.chunk(3,1)

                    sym = torch.round(target_most*(split_scale-1)).reshape(-1).to(torch.int16).to('cpu')  

                    weights = torch.softmax(weights, dim=1)
                    means = means.contiguous()
                    scales = scales.contiguous()
                    weights = weights.contiguous()
                    space_strings.append(torchac.encode_logistic_mixture(sym_value, means, scales, weights, sym))
                    
                    target_least = torch.round(target*(white-black)) % split_scale / (split_scale-1)
                    bit_context = self.bit_trans[i][scale-1](target_most)
                    least_params = self.aggs_least[i][scale-1][space-1](torch.concat([support_context, bit_context], dim=1))
                    means, scales, weights = least_params.chunk(3,1)

                    sym = torch.round(target_least*(split_scale-1)).reshape(-1).to(torch.int16).to('cpu')  
                    weights = torch.softmax(weights, dim=1)
                    means = means.contiguous()
                    scales = scales.contiguous()
                    weights = weights.contiguous()
                    space_strings.append(torchac.encode_logistic_mixture(sym_value, means, scales, weights, sym))
                    space_support = torch.concat([space_support, target], dim=1)   
                
                space_support = self.merge_scale(space_support)             
                
                
            x_strings.append(space_strings)                
        return{
            'shapes':[x.shape, out_latent_compress['shape']],
            'strings':[out_latent_compress['strings'], x_strings],
        }

    #single image decompress
    def decompress(self, strings, shapes, black:int, white:int):
        x_shape = shapes[0]
        latent_strings = strings[0]
        x_strings = strings[1]
        
        B,C, H, W = x_shape
        total_scale = self.scale
        prior = self.latent_compress.decompress(latent_strings, shapes[1])[:,:,:H,:W]
        
        split_scale = 2**(self.bit//2)
        bin = 1/(split_scale-1)
        sym_value = torch.linspace(bin / 2, 1 - bin / 2, split_scale-1, dtype=torch.float32, device=prior.device)  
        value_min = torch.tensor([-1], device=prior.device)
        value_max = torch.tensor([2], device=prior.device)
        sym_value = torch.concatenate([value_min, sym_value, value_max], dim=0)

        x_decoded = torch.zeros([B,C,H,W], device=prior.device)
        
        for i in range(4):
            if i == 0:
                color_context = prior
            else:
                channel_context = self.color_trans[i-1](x_decoded[:,:i,:,:])
                color_context = torch.concat([prior, channel_context], dim=1)
            
            space_strings = x_strings[i]
            support_context = color_context[:, :, 0::(2**total_scale), 0::(2**total_scale)]  
            most_params = self.aggs_first_most[i](support_context)
            means, scales, weights = most_params.chunk(3,1)
            weights = torch.softmax(weights, dim=1)
            means = means.contiguous()
            scales = scales.contiguous()
            weights = weights.contiguous()
            
            target_most = torchac.decode_logistic_mixture(sym_value, means, scales, weights, space_strings[0]).to(support_context.device).reshape(B, 1, H//(2**total_scale), W//(2**total_scale))/(split_scale-1)
            bit_context = self.bit_trans[i][total_scale-1](target_most)
            least_params = self.aggs_first_least[i](torch.concat([support_context, bit_context], dim=1))
            means, scales, weights = least_params.chunk(3,1)
            weights = torch.softmax(weights, dim=1)
            means = means.contiguous()
            scales = scales.contiguous()
            weights = weights.contiguous()
            target_least = torchac.decode_logistic_mixture(sym_value, means, scales, weights, space_strings[1]).to(support_context.device).reshape(B, 1, H//(2**total_scale), W//(2**total_scale))/(split_scale-1)
            target = (torch.round(target_most*(split_scale-1)*split_scale + target_least*(split_scale-1))) / white
            x_decoded[:, i:i+1, 0::(2**total_scale), 0::(2**total_scale)] = target

            space_support = target
            count = 1
            
            for scale in reversed(range(1, total_scale+1)):
                step = 2**scale
                for space in range(1, 4):                    
                    support_latent = color_context[:, :, (space//2)*(step//2)::step, (space%2)*(step//2)::step]
                    space_context = self.spatial_trans[i][scale-1][space-1](space_support)    
                    support_context = torch.concat([support_latent, space_context], dim=1)       
                    
                    most_params = self.aggs_most[i][scale-1][space-1](support_context)
                    means, scales, weights = most_params.chunk(3,1)
                    weights = torch.softmax(weights, dim=1)
                    means = means.contiguous()
                    scales = scales.contiguous()
                    weights = weights.contiguous()
                    target_most = torchac.decode_logistic_mixture(sym_value, means, scales, weights, space_strings[2*count]).to(support_context.device).reshape(B, 1, H//step, W//step)/(split_scale-1)
                    
                    bit_context = self.bit_trans[i][scale-1](target_most)
                    least_params = self.aggs_least[i][scale-1][space-1](torch.concat([support_context, bit_context], dim=1))
                    means, scales, weights = least_params.chunk(3,1)
                    weights = torch.softmax(weights, dim=1)
                    means = means.contiguous()
                    scales = scales.contiguous()
                    weights = weights.contiguous()
                    target_least = torchac.decode_logistic_mixture(sym_value, means, scales, weights, space_strings[2*count+1]).to(support_context.device).reshape(B, 1, H//step, W//step)/(split_scale-1)
                    target = (torch.round(target_most*(split_scale-1)*split_scale + target_least*(split_scale-1))) / white
                    count += 1
                    space_support = torch.concat([space_support, target], dim=1)   
                    x_decoded[:, i:i+1, (space//2)*(step//2)::step, (space%2)*(step//2)::step] = target
                
                space_support = self.merge_scale(space_support)             

        x_decoded = x_decoded * (white-black) + black
        x_decoded = self.unpack_raw(x_decoded)
        return x_decoded
    

    def update(self):
        self.latent_compress.update()


    def aux_loss(self):
        return self.latent_compress.aux_loss()
