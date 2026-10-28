import struct
import numpy as np

from .utils import *
from .model import ModelInfo
from .layers import ResidualAddInfo

def shape_to_str(s):
    try:
        r = ""
        for v in s:
            r += f"{v}, "
        return r[:-2]
    
    except:
        return f"{s}"

def classic_tensor_buffers(model):
    r = ""
    b = bytearray()

    for l in model.layers:
        # Ignore if no buffers exist
        if l.mu_buffer is None:
            continue

        if model.floating_point:
            strfmt = "f"
            wmu = l.mu_buffer.flatten()
            bmu = l.mu_bias.flatten()
        else:
            strfmt = "i"
            wmu = to_fixed(l.mu_buffer.flatten(), model.fixed_bits)
            bmu = to_fixed(l.mu_bias.flatten(), model.fixed_bits)

        r += f"TensorBuffer<wdt, {shape_to_str(l.mu_buffer.shape)}> {l.name}_weight;\n"
        r += f"TensorBuffer<wdt, {shape_to_str(l.mu_bias.shape)}> {l.name}_bias;\n\n"

        for v in wmu:
            b += bytearray(struct.pack(strfmt, v))
        for v in bmu:
            b += bytearray(struct.pack(strfmt, v))

    r += "void load_weights(const std::string& filename) {\n"
    r += "std::ifstream f;\nf.open(filename);\n\n"
    for l in model.layers:
        # Ignore if no buffers exist
        if l.mu_buffer is None:
            continue
        
        r += f"{l.name}_weight.read_from_file(f);\n"
        r += f"{l.name}_bias.read_from_file(f);\n\n"
    r += "f.close();\n}\n\n"

    return r, b

def create_tensor_buffers(model):
    r = ""
    b = bytearray()

    for l in model.layers:
        # Ignore if no buffers exist
        if l.mu_buffer is None:
            continue

        if model.floating_point:
            strfmt = "ff"
            wmu = l.mu_buffer.flatten()
            wsi = l.sigma_buffer.flatten()
            bmu = l.mu_bias.flatten()
            bsi = l.sigma_bias.flatten()
        else:
            strfmt = "ii"
            wmu = to_fixed(l.mu_buffer.flatten(), model.fixed_bits)
            wsi = to_fixed(l.sigma_buffer.flatten(), model.fixed_bits)
            bmu = to_fixed(l.mu_bias.flatten(), model.fixed_bits)
            bsi = to_fixed(l.sigma_bias.flatten(), model.fixed_bits)

        r += f"TensorBuffer<wdt, {shape_to_str(l.mu_buffer.shape)}> {l.name}_weight;\n"
        r += f"TensorBuffer<wdt, {shape_to_str(l.mu_bias.shape)}> {l.name}_bias;\n\n"

        for v in zip(wmu, wsi):
            b += bytearray(struct.pack(strfmt, v[0], v[1]))
        for v in zip(bmu, bsi):
            b += bytearray(struct.pack(strfmt, v[0], v[1]))

    r += "void load_weights(const std::string& filename) {\n"
    r += "std::ifstream f;\nf.open(filename);\n\n"
    for l in model.layers:
        # Ignore if no buffers exist
        if l.mu_buffer is None:
            continue
        
        r += f"{l.name}_weight.read_from_file(f);\n"
        r += f"{l.name}_bias.read_from_file(f);\n\n"
    r += "f.close();\n}\n\n"

    return r, b

def classic_codegen(model_info: ModelInfo, data = None):
    n = model_info.name

    funcs = "#include \"tensorlib/bnn.hpp\"\n\n"
    funcs += f"class Classic_nn {{\n\n"
    funcs += "public:\n"

    if model_info.floating_point:
        funcs += f"using dt = f32;\n"
    else:
        funcs += f"using dt = fxp<i32,i64,{model_info.fixed_bits}>;\n"

    funcs += f"using wdt = dt;\n"
    funcs += f"using adt = dt;\n"
    funcs += "\n"
    funcs += f"using input_t = TensorView<adt, {shape_to_str(model_info.layers[0].in_buffer_shape)}>;\n"
    funcs += f"using out_t = TensorView<adt, {shape_to_str(model_info.layers[-1].output_shape())}>;\n"
    funcs += "\n"

    # Intermediate Buffers
    for i in range(3):
        funcs += f"adt ibuf{i}[{model_info.buffer_max_sizes[i]}];\n"
    funcs += "\n"

    tb, weight_bytes = classic_tensor_buffers(model_info)
    funcs += tb

    funcs += f"void forward(input_t& x) {{\n"
    funcs += "using namespace Tensorlib; using namespace Tensorlib::Bnn;\n\n"

    for i, l in enumerate(model_info.layers):
        lnm = l.name

        if l.is_input:
            funcs += f"input_t {lnm}_in(x);\n"
        else:
            funcs += f"TensorView<adt, {shape_to_str(l.in_buffer_shape)}> "
            funcs += f"{lnm}_in(ibuf{l.in_addr});\n"

            if isinstance(l, ResidualAddInfo):
                funcs += f"TensorView<adt, {shape_to_str(l.in_buffer_shape)}> "
                funcs += f"{lnm}_in_res(ibuf{l.residual_addr});\n"

        funcs += f"TensorView<adt, {shape_to_str(l.output_shape())}> "
        funcs += f"{lnm}_out(ibuf{l.out_addr});\n"
        funcs += l.cppfcall(model_info)

    funcs += "}\n\n"

    if data is not None:
        funcs += f"TensorBuffer<adt, {shape_to_str(data.shape)}> test_data;\n"
        funcs += "input_t get_data(u32 i) { return input_t(test_data, {i}); }\n"
        funcs += "void load_test_data(const std::string& filename) {\n"
        funcs += "std::ifstream f;\nf.open(filename);\n"
        funcs += "test_data.read_from_file(f);\n"
        funcs += "f.close();\n}\n\n"

        if model_info.floating_point:
            strfmt = "f"
            d = data.flatten().detach().numpy()
        else:
            strfmt = "i"
            d = to_fixed(data.flatten().detach().numpy(), model_info.fixed_bits)

        data_bytes = bytearray()
        for v in d:
            data_bytes += bytearray(struct.pack(strfmt[0], v))
        
    else:
        data_bytes = None

    funcs += "out_t output() { "
    funcs += f"return out_t(ibuf{model_info.layers[-1].out_addr});"
    funcs += " }\n\n"

    funcs += "};\n"

    return funcs, weight_bytes, data_bytes


## Add intermediate mode output
def codegen(model_info: ModelInfo, data = None, split_head = False):
    n = model_info.name

    funcs = "#include \"tensorlib/bnn.hpp\"\n\n"
    funcs += f"class Bnn_{n} {{\n\n"
    funcs += "public:\n"
    funcs += f"static const u32 mc_samples = {model_info.mc_passes};\n"
    funcs += f"static const auto genf = Tensorlib::Bnn::{model_info.get_gen_mode_cppstr()};\n\n"

    if model_info.floating_point:
        funcs += f"using dt = f32;\n"
    else:
        funcs += f"using dt = fxp<i32,i64,{model_info.fixed_bits}>;\n"

    if model_info.inference_mode == InferenceMode.sampling:
        funcs += f"using wdt = Tensorlib::Bnn::Data<dt, Tensorlib::Bnn::Mode::Sample, Tensorlib::Bnn::Family::Uniform>;\n"
        funcs += f"using adt = dt;\n"
    else:
        funcs += f"using wdt = Tensorlib::Bnn::DMPG<dt>;\n"
        funcs += f"using adt = Tensorlib::Bnn::DMPG<dt>;\n"
    
    funcs += "\n"
    if split_head:
        funcs += f"using input_t = TensorView<Tensorlib::Bnn::DMPG<dt>, {shape_to_str(model_info.layers[0].in_buffer_shape)}>;\n"
    else:
        funcs += f"using input_t = TensorView<adt, {shape_to_str(model_info.layers[0].in_buffer_shape)}>;\n"
    funcs += f"using out_t = TensorView<adt, {shape_to_str(model_info.layers[-1].output_shape())}>;\n"
    funcs += "\n"

    # Intermediate Buffers
    for i in range(3):
        funcs += f"adt ibuf{i}[{model_info.buffer_max_sizes[i]}];\n"
    funcs += "\n"

    tb, weight_bytes = create_tensor_buffers(model_info)
    funcs += tb

    funcs += f"void forward(input_t& x) {{\n"
    funcs += "using namespace Tensorlib; using namespace Tensorlib::Bnn;\n\n"
    
    if split_head:
        funcs += f"TensorBuffer<wdt, {shape_to_str(model_info.layers[0].in_buffer_shape)}> mp2s_out;\n"
        funcs += "mp2s_layer(x, mp2s_out);\n\n"
    
    if model_info.inference_mode == InferenceMode.sampling:
        funcs += f"for (u32 i = 0; i < mc_samples; i++) {{\n\n"

    for i, l in enumerate(model_info.layers):
        lnm = l.name

        if l.is_input:
            if split_head:
                funcs += f"TensorView<wdt, {shape_to_str(model_info.layers[0].in_buffer_shape)}>"
                funcs += f"{lnm}_in(mp2s_out);\n"
            else:
                funcs += f"input_t {lnm}_in(x);\n"
        else:
            funcs += f"TensorView<adt, {shape_to_str(l.in_buffer_shape)}> "
            funcs += f"{lnm}_in(ibuf{l.in_addr});\n"

            if isinstance(l, ResidualAddInfo):
                funcs += f"TensorView<adt, {shape_to_str(l.in_buffer_shape)}> "
                funcs += f"{lnm}_in_res(ibuf{l.residual_addr});\n"

        funcs += f"TensorView<adt, {shape_to_str(l.output_shape())}> "
        funcs += f"{lnm}_out(ibuf{l.out_addr});\n"
        funcs += l.cppfcall(model_info)

    if model_info.inference_mode == InferenceMode.sampling:
        funcs += "}\n"
    funcs += "}\n\n"

    if data is not None:
        funcs += f"TensorBuffer<adt, {shape_to_str(data.shape)}> test_data;\n"
        funcs += "input_t get_data(u32 i) { return input_t(test_data, {i}); }\n"
        funcs += "void load_test_data(const std::string& filename) {\n"
        funcs += "std::ifstream f;\nf.open(filename);\n"
        funcs += "test_data.read_from_file(f);\n"
        funcs += "f.close();\n}\n\n"

        if model_info.floating_point:
            strfmt = "ff"
            d = data.flatten().detach().numpy()
        else:
            strfmt = "ii"
            d = to_fixed(data.flatten().detach().numpy(), model_info.fixed_bits)

        data_bytes = bytearray()
        if model_info.layers[0].inference_mode == InferenceMode.sampling:
            for v in d:
                data_bytes += bytearray(struct.pack(strfmt[0], v))
        else:
            for v in d:
                data_bytes += bytearray(struct.pack(strfmt, v, 0))
        
    else:
        data_bytes = None

    funcs += "out_t output() { "
    funcs += f"return out_t(ibuf{model_info.layers[-1].out_addr});"
    funcs += " }\n\n"

    funcs += "};\n"

    return funcs, weight_bytes, data_bytes
    

