import os
import time
from pandas import read_csv
from multiprocessing import Pool, Process

import torch

import numpy as np
import numpy.typing as npt

import empbnn
from empbnn.utils import *
from empbnn.codegen import codegen

import testconf


def parse_predictions(path):
    with open(path, "r") as f:
        nsamples, ndata = map(int, f.readline().split())
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data.reshape((ndata, nsamples, data.shape[1]))


def cpu_inf(work):
    i, model_info, data = work
    code, wbytes, dbytes = empbnn.codegen.codegen(model_info, data)
    testconf.write_c_model(f"Code/worker_{i}", code, wbytes, dbytes)
    os.system(f"""
        cd Code/worker_{i}
        make
    """)
    return parse_predictions(f"Code/worker_{i}/run_cpp.log")


def cpu_inf_split(work):
    i, modelname, data, nmc = work
    testconf.split_c_model(
        f"Code/worker_{i}", modelname, data,
        genmode=empbnn.utils.Distributions.uniform,
        use_fp=False, num_mc=nmc
    )
    os.system(f"""
        cd Code/worker_{i}
        make split_run
    """)
    return parse_predictions(f"Code/worker_{i}/run_cpp.log")


def test(id, worker_pool, num_workers, nmc):
    os.system("mkdir -p Predictions/cpu")

    for modelname in ["SV"] + ["LENET", "B2N2", "RESNET"]:

        print(modelname)
        
        model_info, data = testconf.get_c_model(
            modelname, genmode=empbnn.utils.Distributions.uniform,
            use_fp=False, use_mp=False, use_simple=False, num_mc=nmc
        )
        work = [(i, model_info, d) for i, d in enumerate(data.chunk(num_workers))]
        results = np.concatenate(worker_pool.map(cpu_inf, work))
        np.savez_compressed(f"Predictions/cpu/{id}_{nmc}_{modelname}_baseline.npz", results)


        model_info, data = testconf.get_c_model(
            modelname, genmode=empbnn.utils.Distributions.uniform,
            use_fp=False, use_mp=True, use_simple=False, num_mc=nmc
        )
        work = [(i, model_info, d) for i, d in enumerate(data.chunk(num_workers))]
        results = np.concatenate(worker_pool.map(cpu_inf, work))
        np.savez_compressed(f"Predictions/cpu/{id}_{nmc}_{modelname}_mp.npz", results)


        model_info, data = testconf.get_c_model(
            modelname, genmode=empbnn.utils.Distributions.uniform,
            use_fp=False, use_mp=True, use_simple=True, num_mc=nmc
        )
        work = [(i, model_info, d) for i, d in enumerate(data.chunk(num_workers))]
        results = np.concatenate(worker_pool.map(cpu_inf, work))
        np.savez_compressed(f"Predictions/cpu/{id}_{nmc}_{modelname}_simple.npz", results)


        data = testconf.get_c_data(modelname)
        work = [(i, modelname, d, nmc) for i, d in enumerate(data.chunk(num_workers))]
        results = np.concatenate(worker_pool.map(cpu_inf_split, work))
        np.savez_compressed(f"Predictions/cpu/{id}_{nmc}_{modelname}_split.npz", results)


if __name__ == "__main__":
    num_workers = 25
    worker_pool = Pool(num_workers)
    for i in range(5):
        print(i + 1)
        test(i + 1, worker_pool, num_workers, 100)
