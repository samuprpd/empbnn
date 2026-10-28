import numpy as np
import numpy.typing as npt

from math import sqrt

from prettytable import PrettyTable, TableStyle

from .layers import *
from .utils import *

import copy

class ModelInfo:
    def __init__(self, name: str):
        self.name = name

        # Default init values
        self.layers: list[LayerInfo] = []

        self.buffer_max_sizes = [1,1,1]
        self.buffer_sequence = []
        self.buffers_calculated = False

        self.inference_mode = InferenceMode.sampling
        
        # C Info
        self.riscv_ext = False
        self.mc_passes = 100
        self.gen_mode = Distributions.gaussian
        self.floating_point = True
        self.fixed_bits = 12

        self.data_types = {
            "MU":           CTypes.i32,
            "SIGMA":        CTypes.i32,
            "MU_BIAS":      CTypes.i32,
            "SIGMA_BIAS":   CTypes.i32,
            "DATA":         CTypes.i32
        }

    def get_gen_mode_cppstr(self):
        if self.gen_mode == Distributions.gaussian:
            return "Family::Gaussian"
        elif self.gen_mode == Distributions.uniform:
            return "Family::Uniform"

    #
    # Delete variances
    #

    def zero_variances(self):
        for l in self.layers:
            l.zero_variance()

    #
    # Layer folding and pruning
    #

    def to_distribution_inference(self):
        self.inference_mode = InferenceMode.distribution
        # Square sigma weights if distribution mode
        i = 0
        for l in self.layers:
            l.to_distribution_inference()

    def split(self, n):
        tail = ModelInfo("tail")
        tail.layers = copy.deepcopy(self.layers[:-n])

        tail.floating_point = self.floating_point
        tail.mc_passes = self.mc_passes
        tail.fixed_bits = self.fixed_bits

        head = ModelInfo("head")
        head.layers = copy.deepcopy(self.layers[-n:])

        head.floating_point = self.floating_point
        head.mc_passes = self.mc_passes
        head.fixed_bits = self.fixed_bits

        return tail, head

    def set_relu_simple(self):
        for l in self.layers:
            if isinstance(l, ActivationReLU):
                l.simple = True

    def set_pooling_simple(self):
        for l in self.layers:
            if isinstance(l, MaxPool2DInfo):
                l.simple = True

    def optimize_model(self):
        self.fold_batch_norms()
        self.fold_activations(ActivationReLU)
        self.set_residual_io()

    def prune(self):
        aux = []
        for l in self.layers:
            if l.folded:
                continue
            else:
                aux.append(l)
        self.layers = aux

    def fold_batch_norms(self):
        for i, l in enumerate(self.layers):
            if isinstance(l, BatchNorm2DInfo):
                l.apply_batch_norm(self.layers[i-1])
                self.layers[i-1].folded_batch_norm = l
                l.fold()
        self.prune()

    def fold_activations(self, activation_layer_class):
        for i, l in enumerate(self.layers):
            if isinstance(l, activation_layer_class):
                self.layers[i-1].activation = l.string_id()
                l.fold()
        self.prune()

    def set_residual_io(self):
        for i, l in enumerate(self.layers):
            if isinstance(l, FoldableInfo) and l.t == "ResidualBlock":
                self.layers[i-1].output_to_residual = True
                self.layers[i+1].input_from_residual = True
                l.fold()
        self.prune()

        for i, l in enumerate(self.layers):
            if isinstance(l, FoldableInfo) and l.t == "ResidualConv":
                self.layers[i+1].input_from_residual = True
                self.layers[i+1].output_to_residual = True
                l.fold()
        self.prune()

    #
    # Buffer preparation and calculations
    #

    def find_buffer(self, buffers_use, use):
        for i, u in enumerate(buffers_use):
            if u == use:
                return i
        print(f"ERROR NO BUFFERS FREE {buffers_use} -- {use}")
        panic()

    def calculate_buffers(self, input_shape: npt.NDArray):
        # 0 No use
        # 1 Last output
        # 2 Residual
        buffers_use = [0,0,0]
        buffers_shape = [None, None, None]

        for l in self.layers:

            if isinstance(l, ResidualAddInfo):
                o = self.find_buffer(buffers_use, 0)
                i = self.find_buffer(buffers_use, 1)
                r = self.find_buffer(buffers_use, 2)
                l.out_addr = o
                l.in_addr = i
                l.residual_addr = r
                buffers_use[o] = 2 if l.output_to_residual else 1
                buffers_use[i] = 0
                buffers_use[r] = 0
            else:
                # Input
                if l.is_input:
                    i = 0
                    l.in_addr = "data_in"
                    buffers_shape[i] = input_shape
                else:
                    i = self.find_buffer(buffers_use, 2 if l.input_from_residual else 1)
                    l.in_addr = i

                # Output
                o = self.find_buffer(buffers_use, 0)
                l.out_addr = o

                if l.input_from_residual and l.output_to_residual:
                    buffers_use[i] = 0
                    buffers_use[o] = 2
                else:
                    buffers_use[i] = 2 if l.input_from_residual else 0
                    buffers_use[o] = 2 if l.output_to_residual else 1

            # Make sure to flat the dimensions for linear info layers
            if isinstance(l, LinearInfo):
                l.in_buffer_shape = buffers_shape[i].prod()
            else:
                l.in_buffer_shape = buffers_shape[i]

            buffers_shape[o] = l.output_shape()
            self.buffer_max_sizes[o] = max(np.prod(self.buffer_max_sizes[o]), np.prod(buffers_shape[o]))
            self.buffer_sequence.append(str(buffers_use))

        self.buffers_calculated = True

    #
    # Weight transformation optimizations
    #

    def enable_riscv_custom_ext(self):
        self.uniform_weight_transform()
        self.riscv_ext = True

    def uniform_weight_transform(self):
        self.gen_mode = Distributions.uniform

        for l in self.layers:
            if not(isinstance(l, Conv2DInfo) or isinstance(l, LinearInfo)):
                continue
            # W
            l.mu_buffer, l.sigma_buffer = uniform_transform(l.mu_buffer, l.sigma_buffer)
            # B
            l.mu_bias, l.sigma_bias = uniform_transform(l.mu_bias, l.sigma_bias)

    def bernoulli_weight_transform(self):
        self.gen_mode = Distributions.bernoulli

        for l in self.layers:
            if not(isinstance(l, Conv2DInfo) or isinstance(l, LinearInfo)):
                continue
            # W
            l.mu_buffer, l.sigma_buffer = bernoulli_transform(l.mu_buffer, l.sigma_buffer)
            # B
            l.mu_bias, l.sigma_bias = bernoulli_transform(l.mu_bias, l.sigma_bias)

    #
    # Fixed point and data type transformations and calculations
    #

    def find_data_types(self):
        data_range = [None, None, None, None]
        for l in self.layers:
            if not(isinstance(l, Conv2DInfo) or isinstance(l, LinearInfo)):
                continue
    
            # Data range w mu
            data_range[0] = update_data_range(data_range[0], l.mu_buffer)
            # Data range w sigma
            data_range[1] = update_data_range(data_range[1], l.sigma_buffer)
            # Data range b mu
            data_range[2] = update_data_range(data_range[2], l.mu_bias)
            # Data range b sigma
            data_range[3] = update_data_range(data_range[3], l.sigma_bias)
        
        for (min_v, max_v), t in zip(data_range, ["MU", "SIGMA", "MU_BIAS", "SIGMA_BIAS"]):
            if self.floating_point:
                self.data_types[t] = CTypes.f32
            else:
                signed = min_v < 0
                absmax_v = max(abs(min_v), max_v)
                
                if absmax_v == 0:
                    absmax_v = 1
                
                self.data_types[t] = find_ctype(signed, absmax_v)

        if self.floating_point:
            self.data_types["DATA"] = CTypes.f32

    #
    # Output info
    # 

    # Printing and info
    def print_model_stats(self):
        def cost(n):
            return np.array([1, 2*n, n])
        
        r = np.array([0,0,0])
        nw = 0

        for l in self.layers:
            if isinstance(l, LinearInfo):
                s = l.mu_buffer.shape
                r += s[0] * cost(s[1])
                nw += len(l.mu_buffer.flat)
            elif isinstance(l, Conv2DInfo):
                o = l.output_shape().prod()
                ci = l.in_channels
                k = l.kernel_size.prod()
                r += o * cost(ci * k)
                nw += len(l.mu_buffer.flat)
            elif isinstance(l, ResidualAddInfo):
                r += l.output_shape().prod() * cost(1)

        print(f"{r[0]} stores {r[1]} loads {r[2]} ops {nw} weights {sum(self.buffer_max_sizes)} activations")

    def print_info(self):
        t = PrettyTable(border=False, header=False)
        t.set_style(TableStyle.MSWORD_FRIENDLY)
        t.field_names = ["Name", "Type", "Params", "IO", "Inference Mode", "Activation", "BN", "In Shape", "Out Shape" ]
        for _, l in enumerate(self.layers):
            r = l.full_info()
            t.add_row(r)
        print(t)
