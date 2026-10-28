from math import log

import numpy as np
import numpy.typing as npt

import pandas as pd

# Predictionwise Data

def entropy(p: npt.NDArray) -> float:
    eps = np.finfo(np.float64).eps
    h = -np.sum(p * np.log(p + eps))
    return h

def global_uncertainty(p: npt.NDArray) -> float:
    return entropy(p)

def normalized_global_uncertainty(p: npt.NDArray) -> float:
    num_classes = len(p)
    return global_uncertainty(p) / np.log(num_classes)

def expected_entropy(p: npt.NDArray) -> float:  
    return np.array([entropy(pi) for pi in p]).mean()

def normalized_expected_entropy(p: npt.NDArray) -> float:
    num_classes = p.shape[1]
    return expected_entropy(p) / np.log(num_classes)

def mutual_information(p: npt.NDArray, avg: npt.NDArray) -> float:
    return global_uncertainty(avg) - expected_entropy(p)

def normalized_mutual_information(p: npt.NDArray, avg: npt.NDArray) -> float:
    num_classes = len(avg)
    return mutual_information(p, avg)  / np.log(num_classes)

def class_predicted(p: npt.NDArray) -> int:
    return p.argmax()

def confidence(p: npt.NDArray):
    return p.max()

def accurate(p, target):
    return 1 if class_predicted(p) == target else 0


# Error / Datasetwise metrics

def accuracy(data) -> float:
    return data["accurate"].to_numpy().mean()

def brier_score(data):
    nsamples = len(data)
    nclasses = len(data.iloc[0]["average"])

    labels_one_hot = np.zeros((nsamples, nclasses))
    labels_one_hot[range(nsamples), data["target"]] = 1

    preds = np.array([d for d in data["average"]])

    be = ((labels_one_hot - preds) ** 2).sum() / nsamples
    return be
    

def expected_calibration_error(data, nbins=10):
    bsize = 1 / nbins
    bins = np.arange(0, 1 + bsize, bsize)

    nsamples = len(data)
    ece = 0

    confs = data["confidence"].to_numpy()
    accs = data["accurate"].to_numpy()

    for i in range(nbins):
        mask = (confs >= bins[i]) & (confs < bins[i + 1])
        bin_accs = accs[mask]
        bin_confs = confs[mask]

        if len(bin_accs) > 0:
            acc = np.mean(bin_accs)
            conf = np.mean(bin_confs)
            ece += len(bin_accs) * abs(acc - conf) / nsamples
    
    return ece


def adaptive_calibration_error(data, nbins=40):
    nclasses = len(data.iloc[0]["average"])
    asce = 0

    sorted_data = data.sort_values("confidence")

    for c in range(nclasses):
        mask = (sorted_data["target"] == c)
        class_data = sorted_data[mask]
        confs = class_data["confidence"].to_numpy()
        accs = class_data["accurate"].to_numpy()
        bins_confs = np.array_split(confs, nbins)
        bins_accs = np.array_split(accs, nbins)

        for bin_confs, bin_accs in zip(bins_confs, bins_accs):
            if len(bin_confs) > 0:
                acc = np.mean(bin_accs)
                conf = np.mean(bin_confs)
                asce += abs(acc - conf)
    
    asce *= 1 / (nbins * nclasses)
    return asce


def uncertainty_calibration_error(data, nbins=10):
    bsize = 1 / nbins
    bins = np.arange(0, 1 + bsize, bsize)

    nsamples = len(data)
    uce = 0

    confs = data["confidence"].to_numpy()
    accs = data["accurate"].to_numpy()
    uncs = data["uncertainty"].to_numpy()

    for i in range(nbins):
        mask = (confs >= bins[i]) & (confs < bins[i + 1])
        bin_accs = accs[mask]
        bin_uncs = uncs[mask]

        if len(bin_accs) > 0:
            err = 1 - np.mean(bin_accs)
            unc = np.mean(bin_uncs)
            uce += len(bin_accs) * abs(err - unc) / nsamples
    
    return uce


def adaptive_uncertainty_calibration_error(data, nbins=40):
    auce = 0
    sorted_data = data.sort_values("uncertainty")
    
    bins_uncs = np.array_split(sorted_data["uncertainty"].to_numpy(), nbins)
    bins_accs = np.array_split(sorted_data["accurate"].to_numpy(), nbins)

    for bin_uncs, bin_accs in zip(bins_uncs, bins_accs):
        if len(bin_uncs) > 0:
            err = 1 - np.mean(bin_accs)
            unc = np.mean(bin_uncs)
            auce += abs(err - unc)

    auce *= 1 / nbins
    return auce


def ce_loss(prediction, target):
    one_hot = np.zeros_like(prediction)
    one_hot[target] = 1

    # Add smallest float to avoid zeros in log
    eps = np.finfo(np.float64).eps
    lx = np.log(prediction + eps)
    ce = -(lx * one_hot).sum() # Cross Entropy Loss (Softmax + NLL)

    return ce


def analyze_predictions(predictions, targets):
    l = []
    for prediction, target in zip(predictions, targets):
        avg = prediction.mean(axis = 0)

        l.append({
            "prediction": prediction,
            "average": avg,
            "confidence": confidence(avg),
            "class_predicted": class_predicted(avg),
            "uncertainty": normalized_global_uncertainty(avg),
            "expected_entropy": normalized_expected_entropy(prediction),
            "mutual_information": normalized_mutual_information(prediction, avg),
            "target": target,
            "accurate": accurate(avg, target)
        })
    data = pd.DataFrame(l)

    return {
        "analyzed_predictions": data,
        "acc": accuracy(data),
        "ace": adaptive_calibration_error(data),
        "auce": adaptive_uncertainty_calibration_error(data)
    }

def match_ratio(a: npt.NDArray, b: npt.NDArray) -> float:

    match_mask = a["class_predicted"].to_numpy() == b["class_predicted"].to_numpy() 
    diff_mask = ~match_mask

    return np.mean(match_mask), diff_mask


def confidence_quality(data, nths = 50):
    step_size = 1 / nths
    ths = np.arange(step_size, 1 + step_size, step_size)

    r = []

    for i, th in enumerate(ths):

        m = data["confidence"] > th
        acc = data[m]["accurate"].mean()
        cov = len(data[m]) / len(data)

        r.append({
            "ths": th,
            "acc": acc,
            "cover": cov
        })

    return pd.DataFrame(r)


def uncertainty_quality(data, nths = 50):
    nsamples = len(data)

    mask_correct = data["accurate"].to_numpy() == 1
    mask_error = ~mask_correct

    uncs = data["uncertainty"].to_numpy()
    correct_uncs = uncs[mask_correct]
    error_uncs = uncs[mask_error]

    step_size = 1 / nths
    uths = np.arange(step_size, 1 + step_size, step_size)

    r = []

    for i, uth in enumerate(uths):
        nau = (correct_uncs > uth).sum()
        nac = (correct_uncs <= uth).sum()
        nic = (error_uncs <= uth).sum()
        niu = (error_uncs > uth).sum()

        p_acc_cert = nac / (nac + nic)
        p_unc_inn = niu / (nic + niu)
        avuc = (nac + niu) / (nac + nau + nic + niu)

        r.append({
            "uth": uth,
            "p_acc_cert": p_acc_cert,
            "p_unc_inn": p_unc_inn,
            "p_cert": (nac + nic) / nsamples,
            "avuc": avuc
        })

    return pd.DataFrame(r)

def reliability_data(data, nbins = 25, classic = False):
    targets = data["target"].to_numpy()
    averages = np.stack(data["prediction"]) if classic else np.stack(data["average"])
    _, num_classes = averages.shape

    labels_one_hot = np.zeros((len(targets), num_classes))
    labels_one_hot[range(len(targets)), targets] = 1

    # Bins
    bsize = 1 / nbins
    p_groups = np.arange(0, 1 + bsize, bsize)
    centers = p_groups[:-1] + (p_groups[1:] - p_groups[:-1]) / 2

    r = []

    for i in range(len(p_groups) - 1):
        p_min = p_groups[i]
        p_max = p_groups[i + 1]
        group = labels_one_hot[(averages >= p_min) & (averages < p_max)]
        
        if len(group) > 0:
            r.append(group.mean())
        else:
            r.append(0)

    return pd.DataFrame({
        "predicted": centers,
        "observed": r,
    })

# Threshold based metrics

def accuracy_uth(data, uth):
    uncs = data["uncertainty"].to_numpy()
    return accuracy(data[(uncs <= uth)])


def cover(data, uth):
    uncs = data["uncertainty"].to_numpy()
    return (uncs <= uth).sum() / len(uncs)


def avuc(data, uth = 0.8):

    mask_correct = data["accurate"].to_numpy() == 1
    mask_error = ~mask_correct

    uncs = data["uncertainty"].to_numpy()
    correct_uncs = uncs[mask_correct]
    error_uncs = uncs[mask_error]

    nau = (correct_uncs > uth).sum()
    nac = (correct_uncs <= uth).sum()
    nic = (error_uncs <= uth).sum()
    niu = (error_uncs > uth).sum()

    return (nac + niu) / (nac + nau + nic + niu)


def find_best_avuc(data, step = 0.01):
    best = 0
    best_uth = 0

    for uth in np.arange(0.0, 1.0, step):
        s = avuc(data, uth)
        if s > best:
            best = s
            best_uth = uth

    return best_uth


