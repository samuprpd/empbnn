import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.flipout_layers import conv_flipout as bnn_conv, linear_flipout as bnn_linear

from .model import *

class BasicBlock(nn.Module):
    
    def __init__(self, layer: nn.Module, activation: str, flatten: bool = False, pool: nn.Module = None):
        super().__init__()
        self.layer = layer
    
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)

        self.pool = pool

        self.flatten = flatten

    def forward(self, x):

        if self.flatten:
            if len(x.shape) == 4:
                x = x.permute((0,2,3,1)).flatten(1)
            elif len(x.shape) == 5:
                x = x.permute((0,1,3,4,2)).flatten(2)

        x = self.layer(x)
        x = self.activation(x)

        if self.pool is not None:
            x = self.pool(x)

        return x

# Residual block, required for creating the residual buffers

class AddResidual(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        x = x + y
        return x

class ResidualConv(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, x):
        x = self.l(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        if stride == 1:
            c1 = nn.Conv2d(in_channels, out_channels, 3, 1, "same")
        else:
            c1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)

        self.conv1 = c1
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, "same")
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.diff = in_channels != out_channels
        if (self.diff):
            self.extra_conv = ResidualConv(c1)

        self.res = AddResidual()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if(self.diff):
            x = self.extra_conv(x)

        y = self.res(x, y)
        y = self.relu2(y)
        return y

class WideResidualBlock(nn.Module):
    def __init__(self, n, in_channels, out_channels, stride):
        super().__init__()

        l = []
        for i in range(int(n)):
            if i == 0:
                l.append(ResidualBlock(in_channels, out_channels, stride))
            else:
                l.append(ResidualBlock(out_channels, out_channels, 1))
        
        self.layer = nn.Sequential(*l)
    
    def forward(self, x):
        return self.layer(x)

#
# Load model from torch
#

def parse_torch_model(model: nn.Module, name: str):
    """
    Model linear layers must be trained with a flattened input tensor with shape
    [width x height x channels]. Care by default PyTorch layers use shape
    [channels x width x height]
    """

    model_info = ModelInfo(name)
    for n, t in model.named_modules():
        if n == "":
            continue
        l = LayerInfo(n.replace(".", "_"))

        if isinstance(t, nn.MaxPool2d):
            l = MaxPool2DInfo(l, "max", t.kernel_size)

        elif isinstance(t, nn.AvgPool2d):
            l = AvgPool2DInfo(l, "avg", t.kernel_size)

        elif isinstance(t, bnn_conv.Conv2dFlipout):
            l = Conv2DInfo(l)
            
            l.padding = t.padding
            l.kernel_size = np.array(t.kernel_size)
            l.stride = np.array(t.stride)
            l.in_channels = t.in_channels
            l.out_channels = t.out_channels
            # Buffers
            l.mu_buffer = t.mu_kernel.permute((0,2,3,1)).detach().numpy()
            l.sigma_buffer = F.softplus(t.rho_kernel).permute((0,2,3,1)).detach().numpy()
            l.mu_bias = t.mu_bias.detach().numpy()
            l.sigma_bias = F.softplus(t.rho_bias).detach().numpy()
        
        elif isinstance(t, bnn_linear.LinearFlipout):
            l = LinearInfo(l)
            
            l.in_features = t.in_features
            l.out_features = t.out_features
            # Buffers
            l.mu_buffer = t.mu_weight.detach().numpy()
            l.sigma_buffer = F.softplus(t.rho_weight).detach().numpy()
            l.mu_bias = t.mu_bias.detach().numpy()
            l.sigma_bias = F.softplus(t.rho_bias).detach().numpy()

        elif isinstance(t, nn.ReLU):
            l = ActivationReLU(l)
        elif isinstance(t, nn.Softmax):
            l = ActivationSoftmax(l)

        elif isinstance(t, nn.BatchNorm2d):
            l = BatchNorm2DInfo(l)
            l.bn_gamma = t.weight.detach().numpy()
            l.bn_beta = t.bias.detach().numpy()
            l.bn_mean = t.running_mean.detach().numpy()
            l.bn_var = t.running_var.detach().numpy()
            l.bn_eps = t.eps

        elif isinstance(t, ResidualBlock):
            l = FoldableInfo(l, "ResidualBlock")
        elif isinstance(t, ResidualConv):
            l = FoldableInfo(l, "ResidualConv")
        elif isinstance(t, AddResidual):
            l = ResidualAddInfo(l)

        else:
            # Not implemented layers
            continue
        
        model_info.layers.append(l)

    model_info.prune()
    model_info.layers[0].is_input = True
    return model_info

# GPU Moment Propagation layers
# Data type [2, batch, ...]


class LinearMomentProp(nn.Module):
    def __init__(self, original_layer):
        super().__init__()

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        self.w_mu = nn.Parameter(original_layer.mu_weight.clone()) 
        self.w_var = nn.Parameter(F.softplus(original_layer.rho_weight).pow(2))

        self.b_mu = nn.Parameter(original_layer.mu_bias.clone())
        self.b_var = nn.Parameter(F.softplus(original_layer.rho_bias).pow(2))

    def forward(self, x):
        x_mu = x[0,:]
        x_var = x[1,:]

        r_mu = F.linear(x_mu, self.w_mu, self.b_mu)

        r_var = F.linear(x_var, self.w_var, self.b_var)
        r_var += F.linear(x_var, self.w_mu.pow(2))
        r_var += F.linear(x_mu.pow(2), self.w_var)
        # Some cuDNN convolution algorithms return small negative values
        r_var = r_var.clamp(min=0)

        return torch.stack((r_mu, r_var))


class Conv2dMomentProp(nn.Module):
    def __init__(self, original_layer):
        super().__init__()

        self.padding = original_layer.padding
        self.stride = original_layer.stride

        self.w_mu = nn.Parameter(original_layer.mu_kernel.clone())
        self.w_var = nn.Parameter(F.softplus(original_layer.rho_kernel).pow(2))

        self.b_mu = nn.Parameter(original_layer.mu_bias.clone())
        self.b_var = nn.Parameter(F.softplus(original_layer.rho_bias).pow(2))

    def batch_norm_fold(self, bn_layer):
        bn_coef = bn_layer.weight / torch.sqrt(bn_layer.running_var + bn_layer.eps)
        bn_coef_k = bn_coef.view(-1,1,1,1).expand(self.w_mu.shape)

        self.w_mu = nn.Parameter(self.w_mu * bn_coef_k)
        self.w_var = nn.Parameter((torch.sqrt(self.w_var) * bn_coef_k).pow(2))
    
        self.b_mu = nn.Parameter((self.b_mu - bn_layer.running_mean) * bn_coef + bn_layer.bias)
        self.b_var = nn.Parameter((torch.sqrt(self.b_var) * bn_coef).pow(2))

    def forward(self, x):
        x_mu = x[0,:]
        x_var = x[1,:]

        r_mu = F.conv2d(x_mu, self.w_mu, self.b_mu, stride=self.stride, padding=self.padding)
       
        r_var = F.conv2d(x_var, self.w_var, self.b_var, stride=self.stride, padding=self.padding) 
        r_var += F.conv2d(x_var, self.w_mu.pow(2), stride=self.stride, padding=self.padding)
        r_var += F.conv2d(x_mu.pow(2), self.w_var, stride=self.stride, padding=self.padding)
        # Some cuDNN convolution algorithms return small negative values
        r_var = r_var.clamp(min=0)

        return torch.stack((r_mu, r_var))

class BatchNormMomentProp(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.gamma = nn.Parameter(original_layer.weight)
        self.beta = nn.Parameter(original_layer.bias)
        self.running_mean = nn.Buffer(original_layer.running_mean)
        self.running_var = nn.Buffer(original_layer.running_var)
        self.eps = original_layer.eps
        self.momentum = original_layer.momentum

    def forward(self, x):
        x_mu = x[0,:]
        x_var = x[1,:]

        # Defined p(z) = mean(p(x)) across batch dimension
        
        if self.training:
            batch_size = x_mu.shape[0]
            z_mu = x_mu.mean(dim=0)
            z_var = (x_mu.pow(2) + x_var).mean(dim=0) - x_mu.sum(0).pow(2) / (batch_size**2)
            
            # Update running mean and running var using momentum
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * z_mu
            self.running_var = (1 - self.momentum) * self.running_var + self.running_var * z_var
        else:
            z_mu = self.running_mean
            z_var = self.running_var

        r_mu = self.gamma * (x_mu - z_mu) / torch.sqrt(z_var + self.eps) + self.beta
        r_var = self.gamma.pow(2) * x_var / (z_var + self.eps)

        return torch.stack((r_mu, r_var))


class AvgPool2dMomentProp(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
    
    def forward(self, x):
        x_mu = x[0,:]
        x_var = x[1,:]

        r_mu = F.avg_pool2d(x_mu, self.kernel_size, self.stride)
        r_var = F.avg_pool2d(x_var, self.kernel_size, self.stride) / (self.kernel_size**2)

        return torch.stack((r_mu, r_var))


def _online_mean_var(sample_f, nsamples):
    # Online shifted variance calculation (r, r2, k)
    k = sample_f()
    r = torch.zeros_like(k)
    r2 = torch.zeros_like(k)
    
    for _ in range(nsamples - 1):
        x_sample = sample_f()
        r += x_sample - k
        r2 += (x_sample - k).pow(2)

    r_mu = k + r / nsamples
    r_var = (r2 - r.pow(2) / nsamples) / (nsamples - 1)

    return torch.stack((r_mu, r_var))


class MaxPoolMomentProp(nn.Module):
    def __init__(self, original_layer, num_mc=100, use_simple=False, use_large=False):
        super().__init__()
        self.num_mc = num_mc
        self.use_simple = use_simple
        self.use_large = use_large
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
    
    def mc_forward(self, x):
        x_mu = x[0,:]
        x_std = torch.sqrt(x[1,:])

        def _large_f():
            s = torch.empty((self.num_mc, *x_mu[0,:].shape), device=x_mu.device)
            s = s.normal_().unsqueeze(1).expand(self.num_mc, *x_mu.shape)
            s = torch.addcmul(x_mu.unsqueeze(0).expand(self.num_mc, *x_mu.shape), s, x_std.unsqueeze(0).expand(self.num_mc, *x_std.shape))
            
            # Flat the samples and the batch in the same first dimension
            s = s.view(self.num_mc * s.shape[1], *s[0,0,:].shape)
            s = F.max_pool2d(s, self.kernel_size, self.stride)
            # Divide the first 2 dimensions again in samples, batch
            s = s.view(self.num_mc, x_mu.shape[0], *s[0].shape)

            return torch.stack((s.mean(dim=0), s.var(dim=0)))

        # Allocate tensor in device and share samples across batch
        def _batch_f():
            n = torch.empty_like(x_mu[0,:]).normal_().unsqueeze(0).expand(*x_mu.shape)
            s = n * x_std + x_mu
            return F.max_pool2d(s, self.kernel_size, self.stride)
    
        if self.use_large:
            return _large_f()
        else:
            return _online_mean_var(_batch_f, self.num_mc)
    
    def simple_forward(self, x):
        x_mu = x[0,:]
        x_var = x[1,:]

        r_mu, idx = F.max_pool2d(x_mu, self.kernel_size, self.stride, return_indices=True)
        mask = F.max_unpool2d(torch.ones_like(r_mu), idx, self.kernel_size, self.stride)
        r_var = F.max_pool2d(mask * x_var, self.kernel_size, self.stride)

        return torch.stack((r_mu, r_var))

    def forward(self, x):
        if self.use_simple:
            return self.simple_forward(x)
        else:
            return self.mc_forward(x)


class ReLUMomentProp(nn.Module):
    def __init__(self, num_mc=100, use_simple=False, use_large=False):
        super().__init__()
        self.num_mc = num_mc
        self.use_simple = use_simple
        self.use_large = use_large

    def mc_forward(self, x):
        x_mu = x[0,:]
        x_std = torch.sqrt(x[1,:])

        def _large_f():
            r = torch.empty((self.num_mc, *x_mu[0,:].shape), device=x_mu.device)
            r = r.normal_().unsqueeze(1).expand(self.num_mc, *x_mu.shape)
            r = torch.addcmul(x_mu.unsqueeze(0).expand(self.num_mc, *x_mu.shape), r, x_std.unsqueeze(0).expand(self.num_mc, *x_std.shape))
            r = F.relu(r)
            return torch.stack((r.mean(dim=0), r.var(dim=0)))

        def _batch_f():
            n = torch.empty_like(x_mu[0,:]).normal_().unsqueeze(0).expand(*x_mu.shape)
            s = n * x_std + x_mu
            return F.relu(s)
        
        if self.use_large:
            return _large_f()
        else:
            return _online_mean_var(_batch_f, self.num_mc)
    
    def simple_forward(self, x):
        x_mu = x[0,:]
        x_var = x[1,:]

        r_mu = F.relu(x_mu)
        r_var = x_var * (x_mu > 0)

        return torch.stack((r_mu, r_var))

    def forward(self, x):
        if self.use_simple:
            return self.simple_forward(x)
        else:
            return self.mc_forward(x)


class SoftmaxMomentProp(nn.Module):
    def __init__(self, num_mc=100):
        super().__init__()
        self.num_mc = num_mc

    def forward(self, x):
        x_mu = x[0,:]
        x_std = torch.sqrt(x[1,:])

        r = torch.empty((self.num_mc, *x_mu[0,:].shape), device=x_mu.device)
        r = r.normal_().unsqueeze(1).expand(self.num_mc, *x_mu.shape)
        r = torch.addcmul(x_mu.unsqueeze(0).expand(self.num_mc, *x_mu.shape), r, x_std.unsqueeze(0).expand(self.num_mc, *x_std.shape))
        return F.softmax(r, dim=2)


class BnnSampler(nn.Module):
    def __init__(self, layer, num_mc=100, use_large=False):
        super().__init__()
        self.num_mc = num_mc 
        self.l = layer
        self.use_large = use_large

    def forward(self, x):
        x_mu = x[0,:]
        x_std = torch.sqrt(x[1,:])
        
        r = []
        for _ in range(self.num_mc):
            s = torch.addcmul(x_mu, torch.empty_like(x_mu).normal_(), x_std)
            r.append(self.l(s))

        return torch.stack(r)


def bnn_to_mp(m, use_simple, use_large, num_mc):
    prev = None
    for name, t in list(m._modules.items()):
        # Recursive call (modules inside module)
        if m._modules[name]._modules:
            bnn_to_mp(m._modules[name], use_simple, use_large, num_mc)

        if isinstance(t, bnn_linear.LinearFlipout):
            setattr(m, name, LinearMomentProp(t))
        elif isinstance(t, bnn_conv.Conv2dFlipout):
            setattr(m, name, Conv2dMomentProp(t))
        elif isinstance(t, nn.AvgPool2d):
            setattr(m, name, AvgPool2dMomentProp(t))
    
        elif isinstance(t, nn.ReLU):
            l = ReLUMomentProp()
            l.use_large = use_large
            l.use_simple = use_simple
            l.num_mc = num_mc
            setattr(m, name, l)
        elif isinstance(t, nn.MaxPool2d):
            l =  MaxPoolMomentProp(t)
            l.use_large = use_large
            l.use_simple = use_simple
            l.num_mc = num_mc
            setattr(m, name, l)
        elif isinstance(t, nn.Softmax):
            l = SoftmaxMomentProp()
            l.num_mc = num_mc
            setattr(m, name, l)

        elif isinstance(t, nn.BatchNorm2d):
            prev_m, prev_name = prev
            prev_m._modules[prev_name].batch_norm_fold(t)
            setattr(m, name, nn.Identity())

        prev = (m, name)

