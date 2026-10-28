import time

import numpy as np
from prettytable import PrettyTable, TableStyle

import torch

import testconf
import empbnn

# Candidate inference strategies
STRATEGIES = [("mc", "MC baseline"), ("mp", "Hybrid MP"), ("mp simple", "Simple"), ("split", "Split")]

def find_best_strategy(model: str, val_data, targets, max_acc_drop: float = None, max_ace_incr: float = None, max_auce_incr: float = None, num_mc: int = 20, batch_size: int = 256):
    device = testconf.get_device()
    loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    targets = np.asarray(targets)

    report = []
    for mtype, label in STRATEGIES:
        m, _ = testconf.get_model(model, "trained bnn")
        if mtype != "mc":
            m = testconf.change_model_type(model, mtype, m, num_mc)
        m.to(device)
        m.eval()

        preds = []
        times = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                if mtype == "mc":
                    out = torch.stack([m(data) for _ in range(num_mc)])
                else:
                    out = m(torch.stack((data, torch.zeros_like(data))))

                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                preds.append(out.permute((1, 0, 2)).cpu())

        # The first batch includes warm-up costs
        t = np.median(times[1:]) if len(times) > 1 else times[0]
        aop = empbnn.metrics.analyze_predictions(torch.cat(preds).numpy(), targets)
        report.append({"type": mtype, "label": label, "time": t, "acc": 100 * aop["acc"], "ace": 100 * aop["ace"], "auce": 100 * aop["auce"]})

    base = report[0]
    for r in report:
        r["acc_drop"] = base["acc"] - r["acc"]
        r["ace_increase"] = r["ace"] - base["ace"]
        r["auce_increase"] = r["auce"] - base["auce"]
        r["meets"] = ((max_acc_drop is None or r["acc_drop"] <= max_acc_drop)
                      and (max_ace_incr is None or r["ace_increase"] <= max_ace_incr)
                      and (max_auce_incr is None or r["auce_increase"] <= max_auce_incr))

    best = min((r for r in report if r["meets"]), key=lambda r: r["time"])

    table = PrettyTable(["Strategy", "Time (ms)", "Acc (%)", "ACE (%)", "AUCE (%)", "Meets"], border=False, header=False)
    table.set_style(TableStyle.MSWORD_FRIENDLY)
    table.align = "r"
    table.align["Strategy"] = "l"
    for r in report:
        table.add_row([r["label"], f"{1e3 * r['time']:.2f}", f"{r['acc']:.1f}", f"{r['ace']:.1f}", f"{r['auce']:.1f}", "yes" if r["meets"] else "no"])

    print(f"\n{model}: selected {best['label']}")
    print(table)

    return best

## MAIN

if __name__ == "__main__":
    torch.manual_seed(0)
    for m in ["SV", "LENET", "B2N2", "RESNET"]:
        # Validation set: random 10% of the training split
        train_data, _ = testconf.get_data(m)
        idx = torch.randperm(len(train_data))[:len(train_data) // 10].tolist()
        val_data = torch.utils.data.Subset(train_data, idx)
        targets = [train_data.targets[i] for i in idx]
        find_best_strategy(m, val_data, targets, max_acc_drop=3.5, max_ace_incr=3.5, max_auce_incr=3.5, num_mc=20, batch_size=32)
