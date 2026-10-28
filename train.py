import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import empbnn

from math import isnan
import os

from multiprocessing import Pool

class ModelTrainLogger:
    def __init__(self, folder: str, name: str):
        self.best_loss = None
        self.best_acc = None
        self.best_epoch = None

        self.epoch = 1
        self.test_worse = 0

        self.folder = folder
        self.name = name

        self.nan_found = False

    def next_epoch(self):
        self.epoch += 1

    def save_best_model(self, model: nn.Module):
        torch.save(model.state_dict(), f"{self.folder}/best_{self.name}.pth")

    def log_train(self, loss, acc):
        print(f'+ Epoch {self.epoch} Train: Average loss: {loss:.4f}, Accuracy: {100. * acc:.2f}%')

    def log_test(self, model: nn.Module, loss, acc):
        if self.best_acc is None or acc > self.best_acc:
            self.best_loss = loss
            self.best_acc = acc
            self.best_epoch = self.epoch
            self.test_worse = 0
            self.save_best_model(model)
        else:
            self.test_worse += 1

        print(f'- Epoch {self.epoch} Test:  Average loss: {loss:.4f}, Accuracy: {100. * acc:.2f}%')

    def info(self):
        print(f"Test Epoch {self.best_epoch} loss: {self.best_loss:4f}, Accuracy: {100. * self.best_acc:.2f}")

    def is_overfitting(self, threshold: int = 10):
        return self.test_worse > threshold or self.nan_found

    def log_msg(loss, acc):
        print(f'Test: Average loss: {loss:.4f}, Accuracy: {100. * acc:.2f}%')


def get_batch_correct(model_output, target):
    pred = model_output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item()

# Training utils

def train(model: nn.Module, loader, device, optimizer, loss_func, logger: ModelTrainLogger = None):
    model.train()
    train_loss = 0
    train_correct = 0
    nbatches = 0

    for data, target in loader:
        optimizer.zero_grad()
        
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        output = model(data)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        nbatches += 1
        train_correct += get_batch_correct(output, target)

    train_loss /= nbatches
    train_acc = train_correct / len(loader.dataset)
    logger.log_train(train_loss, train_acc)

def bayesian_train(model: nn.Module, loader, device, optimizer, loss_func, num_mc, logger: ModelTrainLogger = None):
    model.train()
    train_loss = 0
    train_correct = 0
    nbatches = 0

    for data, target, in loader:
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        
        output_list = []
        kl_list = []
        for mc_run in range(num_mc):
            output_list.append(model(data))
            kl_list.append(get_kl_loss(model))
        output = torch.mean(torch.stack(output_list), dim=0)
        kl = torch.mean(torch.stack(kl_list), dim=0)
        loss = loss_func(output, target) + (kl / batch_size)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        nbatches += 1
        train_correct += get_batch_correct(output, target)

    train_loss /= nbatches
    train_acc = train_correct / len(loader.dataset)
    logger.log_train(train_loss, train_acc)


def test(model: nn.Module, loader, device, loss_func, logger: ModelTrainLogger = None):
    model.eval()
    test_loss = 0
    test_correct = 0
    nbatches = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)

            test_correct += get_batch_correct(output, target)
            test_loss += loss.item()
            nbatches += 1
    
    test_loss /= nbatches
    test_acc = test_correct / len(loader.dataset)

    if logger is not None:
        logger.log_test(model, test_loss, test_acc)
    else:
        ModelTrainLogger.log_msg(test_loss, test_acc)
    
    return output


def bayesian_test(model: nn.Module, loader, device, loss_func, num_mc, logger: ModelTrainLogger = None):
    model.eval()
    test_loss = 0
    test_correct = 0
    nbatches = 0
    r = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]

            output_list = []
            kl_list = []
            for mc_run in range(num_mc):
                output_list.append(model(data))
                kl_list.append(get_kl_loss(model))
            raw_output = torch.stack(output_list)
            r.append(raw_output.detach().cpu().permute((1,0,2)))
            output = torch.mean(raw_output, dim=0)
            kl = torch.mean(torch.stack(kl_list), dim=0)
            loss = loss_func(output, target) + (kl / batch_size)

            test_correct += get_batch_correct(output, target)
            test_loss += loss.item()
            nbatches += 1
    
    test_loss /= nbatches
    test_acc = test_correct / len(loader.dataset)

    if logger is not None:
        logger.log_test(model, test_loss, test_acc)
    else:
        ModelTrainLogger.log_msg(test_loss, test_acc)

    return torch.cat(r).permute((1,0,2))

def mp_test(model: nn.Module, loader, device, loss_func):
    model.eval()
    test_loss = 0
    test_correct = 0
    nbatches = 0
    r = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            dummy_var = torch.zeros_like(data)
            data = torch.stack((data, dummy_var))

            raw_output = model(data)
            r.append(raw_output.detach().cpu().permute((1,0,2)))

            output = torch.mean(raw_output, dim=0)
            loss = loss_func(output, target)

            test_correct += get_batch_correct(output, target)
            test_loss += loss.item()
            nbatches += 1
    
    test_loss /= nbatches
    test_acc = test_correct / len(loader.dataset)
    ModelTrainLogger.log_msg(test_loss, test_acc)
        
    return torch.cat(r).permute((1,0,2))

import testconf
from dataclasses import dataclass

@dataclass
class TrainParams:
    early_stop = True
    batch_size = 128
    max_epochs = 200

    lr = 0.001
    wd = 0.00001
    
    optim = "Adam"

    sgd_momentum = 0.9

    milestones = [20,40,60,80]
    milestones_gamma = 0.4

    train_mc = 1
    test_mc = 100

    def get_optim_sched(self, model):
        if self.optim == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
            # Dummy scheduler, step on 900000 steps that should not be reached
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 900000)
        elif self.optim == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.sgd_momentum)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.milestones_gamma)
        elif self.optim == "Cos":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.sgd_momentum)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return optimizer, scheduler

def train_model_classic(modelname: str, params: TrainParams):
    device = testconf.get_device()
    train_data, test_data = testconf.get_data(modelname)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    model, loss = testconf.get_model(modelname, "untrained")
    logger = ModelTrainLogger("Model", modelname)
    optimizer, scheduler = params.get_optim_sched(model)

    model.to(device)
    for epoch in range(1, params.max_epochs + 1):
        train(model, train_loader, device, optimizer, loss, logger)
        test(model, test_loader, device, loss, logger)
        if params.early_stop and logger.is_overfitting():
            break
        logger.next_epoch()
        scheduler.step()
        print("")

def train_model_bayes(modelname: str, moped: bool, params: TrainParams):
    device = testconf.get_device()
    train_data, test_data = testconf.get_data(modelname)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    if moped:
        model, loss = testconf.get_model(modelname, "moped")
    else:
        model, loss = testconf.get_model(modelname, "untrained bnn")

    logger = ModelTrainLogger("Model", f"bnn_{modelname}")
    optimizer, scheduler = params.get_optim_sched(model)

    model.to(device)
    for epoch in range(1, params.max_epochs + 1):
        bayesian_train(model, train_loader, device, optimizer, loss, params.train_mc, logger)
        bayesian_test(model, test_loader, device, loss, params.test_mc, logger)
        if params.early_stop and logger.is_overfitting():
            break
        logger.next_epoch()
        scheduler.step()
        print("")

def test_bayes_model(modelname:str, num_mc:int):
    print(f"Model: {modelname}")

    device = testconf.get_device()
    _, test_data = testconf.get_data(modelname)
    batch_size = 256
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    model, loss = testconf.get_model(modelname, "trained bnn")
    model.to(device)
    preds = bayesian_test(model, test_loader, device, loss, num_mc).cpu().numpy()
    np.savez(testconf.baseline_path(modelname), preds)

    targets = test_data.targets
    preds = preds.transpose((1,0,2))
    aop = empbnn.metrics.analyze_predictions(preds, targets)


def test_classic_model(modelname):
    print(f"Model: {modelname}")

    device = testconf.get_device()
    _, test_data = testconf.get_data(modelname)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    model, loss = testconf.get_model(modelname, "trained classic")
    model.to(device)
    preds = test(model, test_loader, device, loss).cpu().numpy()
    np.savez(f"Predictions/{modelname}_classic.npz", preds)

## MAINS

def main_test():
    for modelname in testconf.Conf.model_list:
        test_bayes_model(modelname)

def main_train():
    for model in ["LENET", "RESNET", "B2N2"]:
        t = TrainParams()
        t.lr = 0.001
        train_model_classic(model, t)
        t.lr = 0.0001
        train_model_bayes(model, True, t)

    t = TrainParams()
    t.max_epochs = 100
    t.batch_size = 512
    t.lr = 0.05
    t.wd = 0.005
    t.optim = "SGD"
    train_model_classic("WIDE_RESNET", t)
    t.num_mc = 10
    t.wd = 0.0002
    t.early_stop = False
    train_model_bayes("WIDE_RESNET", True, t)

def mp_gpu_model(modelname, mtype, num_mc):
    print(modelname, mtype)
    device = "cuda"
    batch_size = 256

    model, lf = testconf.get_model(modelname, "trained bnn")
    model = testconf.change_model_type(modelname, mtype, model, num_mc)
    model.to(device)

    _, test_data = testconf.get_data(modelname)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    preds = mp_test(model, test_loader, device, lf).cpu().numpy()
    if mtype == "mp simple":
        mtype = "mp_simple"
    np.savez(f"Predictions/{modelname}_{mtype}_{num_mc}.npz", preds)

    targets = test_data.targets
    preds = preds.transpose((1,0,2))
    aop = empbnn.metrics.analyze_predictions(preds, targets)

def mp_test_main():
    for i in range(5):
        for m in  ["SV", "LENET", "B2N2", "RESNET"]:
            for nmc in [2,5,10,15,20,30,50,100]:
                mp_gpu_model(m, "mp", nmc)
                os.system(f"mv Predictions/{m}_mp_{nmc}.npz Predictions/{i}_{m}_mp_{nmc}.npz")
                mp_gpu_model(m, "mp simple", nmc)
                os.system(f"mv Predictions/{m}_mp_simple_{nmc}.npz Predictions/{i}_{m}_mp_simple_{nmc}.npz")
                mp_gpu_model(m, "split", nmc)
                os.system(f"mv Predictions/{m}_split_{nmc}.npz Predictions/{i}_{m}_split_{nmc}.npz")

if __name__ == "__main__":
    # Train
    main_train()
    # Test BNN
    main_test()
    # Test Moment Propagation
    mp_test_main()

