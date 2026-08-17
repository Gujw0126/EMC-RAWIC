import torch
import torch.nn as nn
from torch import Tensor
from compressai.layers import GDN ,conv3x3, subpel_conv3x3, ResidualBlock
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
import math

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

    
# modified from https://github.com/BYchao100/Deep-Lossy-Plus-Residual-Coding/blob/main/DLPR_ll/ll_model.py 

class AnalysisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=2)
        self.gdn = GDN(out_ch)
        self.skip = conv3x3(in_ch, out_ch, stride=2)
        self.rb = ResidualBlock(out_ch, out_ch)
        
    def forward(self, input):
        out = self.conv1(input)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out = out + self.skip(input)

        out = self.rb(out)
        return out
        

class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rb = ResidualBlock(in_ch, out_ch)
        self.conv_up = subpel_conv3x3(out_ch, out_ch, r=2)
        self.igdn = GDN(out_ch, inverse=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.upsample = subpel_conv3x3(out_ch, out_ch, r=2)
                
    def forward(self, input):
        out1 = self.rb(input)
        
        out = self.conv_up(out1)
        out = self.igdn(out)
        out = self.conv(out)
        out = self.leaky_relu(out)
        
        out = out + self.upsample(out1)
        
        return out
    

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            AnalysisBlock(in_ch, out_ch),
            AnalysisBlock(out_ch, out_ch),
            AnalysisBlock(out_ch, out_ch),
            conv3x3(out_ch, out_ch, stride=2),
        )
        
    def forward(self, input):
        out = self.layers(input)
        return out
    
    
class Decoder(nn.Module):
    def __init__(self, in_ch, prior_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            SynthesisBlock(in_ch, in_ch),
            SynthesisBlock(in_ch, in_ch),
            SynthesisBlock(in_ch, in_ch)
        )
        self.conv_prior = subpel_conv3x3(in_ch, prior_ch, r=2)

    def forward(self, input):
        out = self.layers(input)
        prior = self.conv_prior(out)
        return prior
            

class HyperEncoder(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv3x3(num_ch, num_ch, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(num_ch, num_ch, stride=2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch)
        )
        
    def forward(self, input):
        out = self.layers(input)
        return out
    
    
class HyperDecoder(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers_mu = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(num_ch, num_ch, r=2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(num_ch, num_ch, r=2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

        self.layers_sigma = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(num_ch, num_ch, r=2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(num_ch, num_ch, r=2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )
        
    def forward(self, input):
        mu = self.layers_mu(input)
        sigma = self.layers_sigma(input)
        return mu, sigma


class ResBlock_1x1(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        out = self.layers(input)
        out = out + input

        return out


class ParamAgg(nn.Module):
    def __init__(self, num_ch, num_mixtures, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ResBlock = ResBlock_1x1(num_ch)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, 3*num_mixtures),
        )
         
    def forward(self, input):
        out = self.ResBlock(input)
        out = self.layers(out)
        return out


class ParamAgg2(nn.Module):
    def __init__(self, num_ch, num_mixtures, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ResBlock = ResBlock_1x1(num_ch)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, 12*num_mixtures),
        )
         
    def forward(self, input):
        out = self.ResBlock(input)
        out = self.layers(out)
        return out
    

class LatentCompressor(CompressionModel):
    def __init__(self,in_ch, num_ch, context_dim=256, *args, **kwargs):
        super().__init__(entropy_bottleneck_channels=num_ch, *args, **kwargs)
        self.encoder = Encoder(in_ch, num_ch)
        self.decoder = Decoder(num_ch, context_dim)
        self.hyperencoder = HyperEncoder(num_ch)
        self.hyperdecoder = HyperDecoder(num_ch)
        
        self.gaussian_conditional = GaussianConditional(None)
    
    def forward(self, input):
        y = self.encoder(input)
        z = self.hyperencoder(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        _, y_likelihoods = self.gaussian_conditional(y, sigma_hat, means=mu_hat)
        y_hat = quantize_ste(y - mu_hat) + mu_hat
        prior = self.decoder(y_hat)
        
        return {
            "prior":prior,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }


    def compress(self, x):
        y = self.encoder(x)
        z = self.hyperencoder(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(sigma_hat)
        y_string = self.gaussian_conditional.compress(y, indexes, mu_hat)
        y_hat = quantize_ste(y - mu_hat) + mu_hat
        prior = self.decoder(y_hat)
        return {
            "strings":[y_string, z_string],
            "prior":prior,
            "shape":z.shape[-2:]
        }
    

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(sigma_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=mu_hat)
        prior = self.decoder(y_hat)
        return prior


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
