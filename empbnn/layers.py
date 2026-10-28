
# Type anotations that avoid cyclic import
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import ModelInfo

from collections import defaultdict

import numpy as np
import numpy.typing as npt

from .utils import *

class LayerInfo:
    def __init__(self, name: str):
        # Layer Name
        self.name = name

        # Layer Type
        self.is_input = False
        
        # Layer is folded and can be pruned from model
        self.folded = False

        # Activation function String
        self.activation = "identity"

        # Buffers for mu and sigma
        self.mu_buffer = None
        self.sigma_buffer = None
        self.mu_bias = None
        self.sigma_bias = None

        # Info for convolutional type layers
        self.folded_batch_norm = None

        # Buffer 
        self.in_buffer_shape = None
        self.in_addr = None
        self.out_addr = None

        # Residual I/O
        self.output_to_residual = False
        self.input_from_residual = False

        # Forward execution mode
        self.inference_mode = InferenceMode.sampling

    # Fixed point encoding

    def to_fixed(self, fixed_bits):
        # By default this does nothing
        return
    
    def sigma_to_var(self):
        # Only implemented by layers that have weights
        return

    def to_distribution_inference(self):
        self.inference_mode = InferenceMode.distribution
        self.sigma_to_var()

    # C Code generation

    def layer_id(self, m: ModelInfo):
         return f"{m.name}_{self.name}"

    # Output shape size

    def output_shape(self):
        print(f"ERR Shape Not implemented for {type(self)}")
        panic()

    def activation_buffer_required(self):
        return np.prod(self.in_buffer_shape) + np.prod(self.output_shape())

    def zero_variance(self):
        return

    # Prunning and folding

    def fold(self):
        self.folded = True



    # Printing

    def _info(self):
        return [f"? {self.name} {type(self)} "]

    def _iobufs(self):
        return f"in={self.in_addr}, out={self.out_addr}"

    def full_info(self):
        r = self._info()
        r.append(self._iobufs())
        r.append(self.inference_mode)

        if self.activation == "mc_relu<genf, mc_samples>":
             r.append("RELU")
        else:
            r.append(self.activation)

        if self.folded_batch_norm is not None:
            r.append("BN Folded")
        else:
            r.append('-')

        r.append(self.in_buffer_shape)
        r.append(self.output_shape())
        return r

#
# Foldable info
#

class FoldableInfo(LayerInfo):
    def __init__(self, l: LayerInfo, t: str):
        super().__init__(l.name)
        self.t = t

#
# Activation function layers
#

class ActivationInfo(LayerInfo):
    def __init__(self, l: LayerInfo):
        super().__init__(l.name)

    def output_shape(self):
        return np.copy(self.in_buffer_shape)


class ActivationReLU(ActivationInfo):
    def __init__(self, l: LayerInfo):
        super().__init__(l)
        self.simple = False

    def string_id(self):
        if self.inference_mode == InferenceMode.distribution:
            return "simple_relu" if self.simple else "mc_relu<genf, mc_samples>"
        else:
            return "relu"

    def _info(self):
        return [self.name, f"RELU", "-"]


class ActivationSoftmax(ActivationInfo):

    def _info(self):
        return [self.name, f"SOFTMAX", "-"]

    def cppfcall(self, m: ModelInfo):
        if self.inference_mode == InferenceMode.distribution:
            return f"""mc_softmax<genf, mc_samples>({self.name}_in);\n\n"""
        else:
            return f"""softmax({self.name}_in, {self.name}_out);\nprint_result({self.name}_out);\n"""

#
# Batch Norm layers
#

class BatchNorm2DInfo(LayerInfo):
    def __init__(self, l: LayerInfo):
        super().__init__(l.name)

        # Buffers for batch norm folding
        self.bn_gamma = None
        self.bn_beta = None
        self.bn_var = None
        self.bn_mean = None
        # Numerical stability constant
        self.bn_eps = None

    def bn_coef(self):
        return self.bn_gamma / np.sqrt(self.bn_var + self.bn_eps)

    def apply_batch_norm(self, l: LayerInfo):
        bn_coef = self.bn_coef()

        for i, coef in enumerate(bn_coef):
            l.mu_buffer[i, :] = coef * l.mu_buffer[i, :]
            l.sigma_buffer[i, :] = coef * l.sigma_buffer[i, :]

        l.mu_bias = bn_coef * (l.mu_bias - self.bn_mean) + self.bn_beta
        l.sigma_bias = bn_coef * l.sigma_bias

#
# Residual Add layers
#

class ResidualAddInfo(LayerInfo):
    def __init__(self, l: LayerInfo):
        super().__init__(l.name)
        self.input_from_residual = True
        self.residual_addr = None

    def output_shape(self):
        return self.in_buffer_shape.copy()

    def cppfcall(self, m: ModelInfo):
        return f"""nn_flat_iter({self.name}_in, {self.name}_in_res, {self.name}_out, add_kernel, {self.activation});\n\n"""

    def _iobufs(self):
        return f"in={self.in_addr}, re={self.residual_addr}, out={self.out_addr}"

    def _info(self):
        return [self.name, f"ADD", "-"]

#
# Convolutional 2D layers
#

# !! WARN only supports square kernels

class Conv2DInfo(LayerInfo):
    def __init__(self, l: LayerInfo):
        super().__init__(l.name)

        # Info for convolutional type layers
        self.stride = None
        self.kernel_size = None
        self.in_channels = None
        self.out_channels = None
        self.padding = None
        self.folded_batch_norm = None

    def to_fixed(self, fixed_bits):
        self.mu_buffer = to_fixed(self.mu_buffer, fixed_bits)
        self.sigma_buffer = to_fixed(self.sigma_buffer, fixed_bits)
        self.mu_bias = to_fixed(self.mu_bias, fixed_bits)
        self.sigma_bias = to_fixed(self.sigma_bias, fixed_bits)

    def sigma_to_var(self):
        self.sigma_buffer = self.sigma_buffer**2
        self.sigma_bias = self.sigma_bias**2

    def output_shape(self):
        dims = 2

        output_shape = self.in_buffer_shape.copy()
        output_shape[dims] = self.out_channels

        if isinstance(self.padding, str):
            
            if self.padding == "valid":
                output_shape[:dims] = ((self.in_buffer_shape[:dims] - self.kernel_size[:dims]) / self.stride[:dims]) + 1
            elif self.padding == "same":
                pass
    
        else:
            self.padding = np.array(self.padding)
            output_shape[:dims] = np.floor(((self.in_buffer_shape[:dims] + 2 * self.padding - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1)

        return output_shape

    def calculate_padding(self):
        
        if not isinstance(self.padding, str):
            return self.padding

        if self.padding == "valid":
            return np.array([0,0])
        elif self.padding == "same":
            output = np.ceil(self.in_buffer_shape[:2]/self.stride)
            return np.clip(((self.stride * (output - 1) - self.in_buffer_shape[:2] + (self.kernel_size - 1) + 1) / 2).astype(int), 0, None)

    def cppfcall(self, m: ModelInfo):
        pdi, pdj = self.calculate_padding()
        stri, strj = self.stride
        return f"""nn_conv2d_iter({self.name}_out, {self.name}_in, {self.name}_weight, {self.name}_bias, {{{stri}, {strj}}}, {{{pdi}, {pdj}}}, mac_kernel, {self.activation});\n\n"""

    def _info(self):
        return [self.name, f"CONV2D", f"k={self.kernel_size} s={self.stride} p={self.calculate_padding()}"]

    def zero_variance(self):
        self.sigma_bias = np.zeros_like(self.sigma_bias)
        self.sigma_buffer = np.zeros_like(self.sigma_buffer)
#
# Linear layers
#

class LinearInfo(LayerInfo):
    def __init__(self, l: LayerInfo):
        super().__init__(l.name)

        # Linear layers info
        self.in_features = None
        self.out_features = None

    def to_fixed(self, fixed_bits):
        self.mu_buffer = to_fixed(self.mu_buffer, fixed_bits)
        self.sigma_buffer = to_fixed(self.sigma_buffer, fixed_bits)
        self.mu_bias = to_fixed(self.mu_bias, fixed_bits)
        self.sigma_bias = to_fixed(self.sigma_bias, fixed_bits)

    def sigma_to_var(self):
        self.sigma_buffer = self.sigma_buffer**2
        self.sigma_bias = self.sigma_bias**2

    def output_shape(self):
        return np.array(self.out_features)

    def cppfcall(self, m: ModelInfo):
        return f"""nn_linear_iter({self.name}_out, {self.name}_in, {self.name}_weight, {self.name}_bias, mac_kernel, {self.activation});\n\n"""

    def _info(self):
        return [self.name, f"LINEAR", "-"]

    def zero_variance(self):
        self.sigma_bias = np.zeros_like(self.sigma_bias)
        self.sigma_buffer = np.zeros_like(self.sigma_buffer)

#
# Pooling 2D layers
#

class Pool2DInfo(LayerInfo):
    def __init__(self, l: LayerInfo, f: str, kernel_size: npt.NDArray):
        super().__init__(l.name)

        # Kernel size of pooling
        self.kernel_size = kernel_size

    def output_shape(self):
        dims = 2 
        output_shape = self.in_buffer_shape.copy()
        output_shape[:dims] = output_shape[:dims] / self.kernel_size
        return output_shape

        return r

class MaxPool2DInfo(Pool2DInfo):
    def __init__(self, l: LayerInfo, f: str, kernel_size: npt.NDArray):
        super().__init__(l, f, kernel_size)
        self.simple = False

    def _info(self):
        stride_i = self.kernel_size
        stride_j = self.kernel_size
        return [self.name, f"MAX POOL2D", f"s=[{stride_i} {stride_j}]"]

    def cppfcall(self, m: ModelInfo):
        if self.inference_mode == InferenceMode.distribution:
            if self.simple:
                return f"""nn_pooling2d_iter<{self.kernel_size}, {self.kernel_size}>({self.name}_out, {self.name}_in, simple_max_pool);\n\n"""
            else:
                return f"""nn_pooling2d_iter<{self.kernel_size}, {self.kernel_size}>({self.name}_out, {self.name}_in, mc_max_pool<genf, mc_samples>);\n\n"""
        else:
            return f"""nn_pooling2d_iter<{self.kernel_size}, {self.kernel_size}>({self.name}_out, {self.name}_in, max_pool);\n\n"""


class AvgPool2DInfo(Pool2DInfo):
    def cppfcall(self, m: ModelInfo):
        return f"""nn_pooling2d_iter<{self.kernel_size}, {self.kernel_size}>({self.name}_out, {self.name}_in, avg_pool);\n\n"""

    def _info(self):
        stride_i = self.kernel_size
        stride_j = self.kernel_size
        return [self.name, f"AVG POOL2D", f"s=[{stride_i} {stride_j}]"]

class Sampler(LayerInfo):
    def output_shape(self):
        return np.copy(self.in_buffer_shape)
    
    def _info(self):
        return [self.name, f"Sampler", "-"]
    
    def cppfcall(self, m: ModelInfo):
        return f"""sample_layer(sampler_in, sampler_out);\n\n"""
