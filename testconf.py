import os

import numpy as np

import torch
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

import models

import empbnn
from empbnn.utils import *
from empbnn.torch import bnn_to_mp, parse_torch_model

class Conf:
    # Try Use GPU
    use_cuda = True

    # BNN conversion hyperparmeters
    bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }

    hyper_model_list = ["BO", "IP", "KSC", "PU", "SV"]

    model_list = hyper_model_list + ["LENET", "B2N2", "RESNET"]

    hyper_models_params = {
        "BO": {"num_in": 145, "num_out": 14},
        "IP": {"num_in": 200, "num_out": 16},
        "KSC": {"num_in": 176, "num_out": 13},
        "PU": {"num_in": 103, "num_out": 9},
        "SV": {"num_in": 204, "num_out": 16},
    }

def init_folders():
    os.system(f"""
        mkdir -p Data
        mkdir -p Code
        mkdir -p Figures
        mkdir -p Model
        mkdir -p Predictions
    """)

def baseline_path(model:str):
    return f"Predictions/{model}_baseline.npz"

def prediction_path(modelname:str, generation_method:str, fixed_bits:int):
    return f"Predictions/{modelname}-{generation_method}-{fixed_bits}.npz"

def get_device():
    use_cuda = torch.cuda.is_available() and Conf.use_cuda
    if use_cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class HyperDataset():
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), self.targets[idx]


def get_data(model:str):

    if model in Conf.hyper_model_list:
        dts = np.load(f"Data/{model}.npz")
        train_data = HyperDataset(dts["train_data"], dts["train_targets"])
        test_data = HyperDataset(dts["test_data"], dts["test_targets"])
        return train_data, test_data

    if model == "WIDE_RESNET":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor()
        ])
        train_data = torchvision.datasets.CIFAR100('Data', train=True, download=True, transform=train_transform)
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_data = torchvision.datasets.CIFAR100('Data', train=False, transform=test_transform)
        return train_data, test_data

    if not model == "RESNET":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    train_data = torchvision.datasets.CIFAR10('Data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10('Data', train=False, transform=transform)

    return train_data, test_data

def get_model(model:str, state:str):

    def nll_loss(x, target):
        # Add numerical stability
        eps = np.finfo(np.float64).eps
        return F.nll_loss(torch.log(x + eps), target, reduction="mean")
    
    lf = None

    if model == "WIDE_RESNET":
        lf = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.07)
    else:
        lf = nll_loss

    m = None

    if model == "RESNET":
        m = models.RESNET_TINY()
    elif model == "WIDE_RESNET":
        m = models.WIDE_RESNET(28, 2, 3, 100)
    elif model == "B2N2":
        m = models.B2N2()
    elif model == "LENET":
        m = models.LENET()
    elif model in Conf.hyper_model_list:
        m = models.HYPER(**Conf.hyper_models_params[model])

    if state == "untrained classic":
        pass

    elif state == "trained classic":
        m.load_state_dict(torch.load(f"Model/best_{model}.pth", weights_only=True))

    elif state == "moped":
        m.load_state_dict(torch.load(f"Model/best_{model}.pth", weights_only=True))
        dnn_to_bnn(m, Conf.bnn_prior_parameters)

    elif state == "untrained bnn":
        dnn_to_bnn(m, Conf.bnn_prior_parameters)

    elif state == "trained bnn":
        dnn_to_bnn(m, Conf.bnn_prior_parameters)
        m.load_state_dict(torch.load(f"Model/best_bnn_{model}.pth"))

        if model == "WIDE_RESNET":
            m = models.WIDE_RESNET_INF(m)
            lf = nll_loss

    return m, lf

def fnamegen(
        modelname:str,
        split_model:bool,
        simple_relu:bool,
        simple_pooling:bool,
        mc_passes:int, 
        generation_method
    ):

    fname = f"{modelname}"

    if split_model:
        fname += "_mix"
    else:
        fname += "_moment"

    if simple_relu:
        fname += "_srelu"
    if simple_pooling:
        fname += "_spool"

    fname += f"_mcp_{mc_passes}"

    if generation_method == Distributions.uniform:
        fname += "_uniform"
    elif generation_method == Distributions.bernoulli:
        fname += "_bernoulli"
    elif generation_method == Distributions.gaussian:
        fname += "_gaussian"

    return fname


def change_model_type(model, mtype, base_model, num_mc=20):
    if mtype == "split":
        if model == "RESNET":
            return models.SPLIT_RESNET_TINY(base_model, num_mc)
        elif model == "WIDE_RESNET":
            return models.WIDE_RESNET_SPLIT(base_model, num_mc)
        elif model == "B2N2":
            return models.SPLIT_B2N2(base_model, num_mc)
        elif model == "LENET":
            return models.SPLIT_LENET(base_model, num_mc)
        elif model in Conf.hyper_model_list:
            return models.SPLIT_HYPER(base_model, num_mc)
    
    elif mtype == "mp":
        bnn_to_mp(base_model, use_simple=False, num_mc=num_mc, use_large=False)
        return base_model

    elif mtype == "mp simple":
        bnn_to_mp(base_model, use_simple=True, num_mc=num_mc, use_large=False)
        return base_model


def get_c_data(modelname):
    _, test_data = get_data(modelname)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
    for data, _ in test_loader:
        continue
    if modelname not in Conf.hyper_model_list:
        data = data.permute((0,2,3,1))
    return data


def get_c_model(modelname, genmode, use_fp, use_mp, use_simple, num_mc):
    model, _ = get_model(modelname, "trained bnn")
    model_info = parse_torch_model(model, "bnn_model")

    data = get_c_data(modelname)
    input_shape = np.array(data[0].shape)

    model_info.floating_point = use_fp
    model_info.mc_passes = num_mc
    if use_mp:
        model_info.to_distribution_inference()
    if use_simple:
        model_info.set_relu_simple()
        model_info.set_pooling_simple()

    model_info.optimize_model()
    if model_info.inference_mode == InferenceMode.sampling:
        if genmode == Distributions.uniform:
            model_info.uniform_weight_transform()
    else:
        model_info.gen_mode = genmode
    model_info.calculate_buffers(input_shape)

    return model_info, data

def get_c_classic_model(modelname, use_fp):
    model, _ = get_model(modelname, "trained bnn")
    model_info = parse_torch_model(model, "bnn_model")

    data = get_c_data(modelname)
    input_shape = np.array(data[0].shape)

    model_info.floating_point = use_fp
    model_info.zero_variances()
    model_info.optimize_model()
    model_info.calculate_buffers(input_shape)

    return model_info, data


def write_c_model(folder, code, wbytes, dbytes):
    os.system(f"""
        rm -rf {folder}
        mkdir -p {folder}
        cp -r empbnn/sources_cpp/* {folder}
    """)

    with open(f"{folder}/model.hpp", "w") as f:
        f.write(code)
    with open(f"{folder}/weights.bin", "wb") as f:
        f.write(wbytes)
    with open(f"{folder}/data.bin", "wb") as f:
        f.write(dbytes)


def split_c_model(tdir, modelname, data, genmode, use_fp, num_mc):
    os.system(f"""
        rm -rf {tdir}
        mkdir -p {tdir}
        cp -r empbnn/sources_cpp/* {tdir}
    """)

    model, _ = get_model(modelname, "trained bnn")
    model_info = parse_torch_model(model, "bnn_model")

    input_shape = np.array(data[0].shape)
    model_info.floating_point = use_fp
    model_info.mc_passes = num_mc
    model_info.gen_mode = genmode
    
    tail, head = model_info.split(2)

    # Tail (model starting layers) optimizations
    tail.to_distribution_inference()
    tail.set_relu_simple()
    tail.set_pooling_simple()
    tail.optimize_model()
    tail.calculate_buffers(input_shape)
    # Create the code
    code, wbytes, dbytes = empbnn.codegen.codegen(tail, data)
    with open(f"{tdir}/model_tail.hpp", "w") as f:
        f.write(code)
    with open(f"{tdir}/tail_weights.bin", "wb") as f:
        f.write(wbytes)
    with open(f"{tdir}/data.bin", "wb") as f:
        f.write(dbytes)

    # Add a sampling layer and optimize
    head.layers.insert(0, empbnn.layers.Sampler("sampler"))
    head.layers[0].is_input = True
    head.optimize_model()
    if genmode == Distributions.uniform:
        head.uniform_weight_transform()
    head.calculate_buffers(tail.layers[-1].output_shape())

    code, wbytes, _ = empbnn.codegen.codegen(head, split_head=True)
    with open(f"{tdir}/model_head.hpp", "w") as f:
        f.write(code)
    with open(f"{tdir}/head_weights.bin", "wb") as f:
        f.write(wbytes)

