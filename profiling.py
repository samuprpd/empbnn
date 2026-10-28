import numpy as np
import pandas as pd

import torch
from torch.profiler import profile, ProfilerActivity, record_function

import testconf
import empbnn

import subprocess
import time
import os
import re

def humanize_time_ns(t, uidx = 3):
    units = ["ns", "us", "ms", "s"]
    i = 0
    while t > 1000 and i < uidx and i < len(units):
        t /= 1000
        i += 1
    return f"{t:.2f}"

def ns_to_us(t):
    return t / 1000

def ns_to_s(t):
    return t / (1000 * 1000 * 1000)

def prof_torch(loader, device, model, is_mp, num_mc):
    i = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            if is_mp:
                dummy_var = torch.zeros_like(data)
                data = torch.stack((data, dummy_var))
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function("model_inference"):
                        model(data)

            else:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function("model_inference"):
                        o = []
                        for _ in range(num_mc):
                            o.append(model(data))

            # Just get the 3rd run, model no longer cold and is not last (i hope)
            if i == 2:
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                prof.export_chrome_trace("torch_trace.json")
                return
            
            i += 1

def get_cuda_counters():
    return (
        torch.cuda.power_draw(), 
        torch.cuda.temperature(), 
        torch.cuda.utilization(), 
        torch.cuda.clock_rate(),
    )


def prof_py(loader, device, model, is_mp, num_mc):
    measures = []
    powers = []
    temps = []
    utilizations = []
    freqs = []
    nruns = 0

    wt_s = time.perf_counter_ns()
    powers.append(torch.cuda.power_draw())

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            if is_mp:
                data = torch.stack((data, torch.zeros_like(data)))
                ts = time.perf_counter_ns()
                o = model(data).to("cpu")
                te = time.perf_counter_ns()
            
            else:
                o = []
                ts = time.perf_counter_ns()
                for _ in range(num_mc):
                    o.append(model(data))
                o = torch.stack(o).to("cpu")
                te = time.perf_counter_ns()
            
            p, t, u, f = get_cuda_counters()
            powers.append(p)
            temps.append(t)
            utilizations.append(u)
            freqs.append(f)

            measures.append(te - ts)

            if nruns > 300:
                break
            nruns += 1

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()

    # Wait and measure until power has settled down
    i = 0
    while True:
        p, t, u, f = get_cuda_counters()
        powers.append(p)
        temps.append(t)
        utilizations.append(u)
        freqs.append(f)

        if p / 1000 < 70 and i >= 5:
            break
        # Power measures are updated every second
        time.sleep(1)
        i += 1

    wt = time.perf_counter_ns() - wt_s

    # Ignore first (cold) and last (small batch)
    avg_time = np.array(measures[1:-1]).mean()
    std_time = np.array(measures[1:-1]).std()
    max_power = np.array(powers).max()
    min_power = np.array(powers).min()
    max_temps = np.array(temps).max()
    min_temps = np.array(temps).min()
    peak_util = np.array(utilizations).max()
    peak_freq = np.array(freqs).max() / 1000

    print(
        len(measures[1:-1]),
        f"{ns_to_us(avg_time):.2f}",
        f"{ns_to_us(std_time):.2f}",
        f"{min_power / 1000 :.2f}",
        f"{max_power / 1000 :.2f}",
        f"{min_temps :.2f}",
        f"{max_temps :.2f}",
        f"{peak_util :.2f}",
        f"{peak_mem :.2f}",
        f"{peak_freq :.2f}",
        f"{ns_to_s(wt):.2f}",
        sep="\t"
    )


def prof(mn, batch_size, num_mc, is_mp, large, simple, split):
    device = "cuda"
    model, _ = testconf.get_model(mn, "trained bnn")

    if split:
        testconf.change_model_type(mn, "split", model, num_mc)
    elif is_mp:
        empbnn.torch.bnn_to_mp(model, simple, large, num_mc)
    
    model.to(device)
    model.eval()

    _, ts = testconf.get_data(mn)
    loader = torch.utils.data.DataLoader(ts, batch_size=batch_size)

    label = f"{mn} "
    
    if split:
        label += "Split"
    elif is_mp:
        label += "Moment Prop "
        label += "Simple" if simple else "Sampling Large Mem" if large else "Sampling Small Mem"
    else:
        label += " BNN"

    print(label, num_mc, batch_size, sep="\t", end="\t")
    prof_py(loader, device, model, is_mp, num_mc)
    #prof_torch(loader, device, model, is_mp, num_mc)


def classic_prof(mn, batch_size):
    device = "cuda"
    model, _ = testconf.get_model(mn, "trained classic")
    
    model.to(device)
    model.eval()

    _, ts = testconf.get_data(mn)
    loader = torch.utils.data.DataLoader(ts, batch_size=batch_size)
    label = mn + " Classic"

    print(label, 1, batch_size, sep="\t", end="\t")
    prof_py(loader, device, model, False, 1)


def perf_cmd(cmd, ntimes = 10):
    result = subprocess.run(f"perf stat -r {ntimes} -e power/energy-pkg/ {cmd}", shell=True, capture_output=True)

    for l in result.stderr.splitlines():
        l = l.decode()
        if "power/energy-pkg/" in l:
            energy = float(l.lstrip().split(" ")[0].replace(",",""))
            m = re.search(r"\+-  ([0-9\.,]+)%", l)
            edev = m.groups()[0]

        elif "time elapsed" in l:
            wall_time = float(l.lstrip().split(" ")[0].replace(",","")) * 1000
            m = re.search(r"\+-  ([0-9\.,]+)%", l)
            wtdev = m.groups()[0]

    return energy, edev, wall_time, wtdev


def cpu_perf(modelname, use_uniform, use_fp, use_mp, use_simple, use_split, num_mc, num_img):
    worker_folder = f"Code/prof_worker"
    g = empbnn.utils.Distributions.uniform if use_uniform else empbnn.utils.Distributions.gaussian
    
    model_info, data = testconf.get_c_model(
        modelname, genmode=g, use_fp=use_fp, 
        use_mp=use_mp, use_simple=use_simple, 
        num_mc=num_mc
    )
    data = data[:num_img]

    if use_split:
        testconf.split_c_model(
            worker_folder, modelname, data,
            genmode=g, use_fp=use_fp, num_mc=num_mc
        )
        os.system(f"""
            cd {worker_folder}
            make split
        """)
    else:
        code, wbytes, dbytes = empbnn.codegen.codegen(model_info, data)
        testconf.write_c_model(worker_folder, code, wbytes, dbytes)
        os.system(f"""
            cd {worker_folder}
            make clang_fast
        """)

    en, edev, wt, wtdev = perf_cmd(f"./{worker_folder}/a.out")

    if use_split:
        t = "Split"
    elif use_simple:
        t = "Simple"
    elif use_mp:
        t = "MP"
    else:
        t = "Baseline"

    print(modelname, t, num_mc, num_img, f"{en:.2f}", edev, f"{wt:.3f}", wtdev, sep="\t")

def classic_cpu_pref(modelname, use_fp, num_img=20):
    worker_folder = f"Code/prof_worker"
    model_info, data = testconf.get_c_classic_model(modelname, use_fp=use_fp)
    data = data[:num_img]

    code, wbytes, dbytes = empbnn.codegen.classic_codegen(model_info, data)
    
    testconf.write_c_model(worker_folder, code, wbytes, dbytes)
    os.system(f"""
        cd {worker_folder}
        make classic
    """)

    en, edev, wt, wtdev = perf_cmd(f"./{worker_folder}/a.out")

    t = "Classic"
    print(modelname, t, 1, num_img, f"{en:.2f}", edev, f"{wt:.3f}", wtdev, sep="\t")


def prof_testbench_gpu():
    print(
        "label", "num mc", "batch size", "num runs",
        "avg time (us)", "std time (us)", 
        "min power (w)", "peak power (w)",
        "min temp (Cº)", "max temp (Cº)",
        "peak util (%)", "peak mem (GiB)",
        "peak freq (GHz)",
        "wall time (s)",
        sep="\t"
    )

    for num_mc in [10,20,100]:
        for mn in testconf.Conf.model_list:
            for batch_size in [1,32,64,128,256]:
                # prof(mn, batch_size, num_mc, False, False, False, False)
                # prof(mn, batch_size, num_mc, True, False, False, False)
                # prof(mn, batch_size, num_mc, True, True, False, False)
                # prof(mn, batch_size, num_mc, True, False, True, False)
                prof(mn, batch_size, num_mc, True, False, True, True)

def prof_testbench_gpu_classic():
    print("label", "num mc", "batch size", "num runs",
        "avg time (us)", "std time (us)", 
        "min power (w)", "peak power (w)",
        "min temp (Cº)", "max temp (Cº)",
        "peak util (%)", "peak mem (GiB)",
        "peak freq (GHz)",
        "wall time (s)",
        sep="\t"
    )
    
    for mn in ["SV", "LENET", "B2N2", "RESNET"]:
        classic_prof(mn, 32)


def prof_testbench_cpu():
    nmc = 100

    print("model", "type", "num mc", "num img", "en J", "en dev %", "wt s", "wt dev %", sep="\t")
    for modelname in ["SV", "LENET", "B2N2", "RESNET"]:
        classic_cpu_pref(modelname, use_fp=False, num_img=20)
        cpu_perf(modelname, use_uniform=False, use_fp=False, use_mp=False, use_simple=False, use_split=False, num_mc=nmc, num_img=20)
        cpu_perf(modelname, use_uniform=True, use_fp=False, use_mp=True, use_simple=False, use_split=False, num_mc=nmc, num_img=20)
        cpu_perf(modelname, use_uniform=True, use_fp=False, use_mp=True, use_simple=True, use_split=False, num_mc=nmc, num_img=20)
        cpu_perf(modelname, use_uniform=True, use_fp=False, use_mp=True, use_simple=True, use_split=True, num_mc=nmc, num_img=20)

if __name__ == "__main__":
    for i in range(5):
        prof_testbench_gpu_classic()
