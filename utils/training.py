import os
from tqdm import tqdm
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch import optim
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import losses, miners

from utils.testing import evaluate


def epoch_end(epoch, train_loss, test_loss, train_acc, test_acc):
    print(f'Epochs [{epoch}]:')
    print(f'train loss: {train_loss}; test loss: {test_loss}')
    print(f'train acc: {train_acc}; test acc: {test_acc}')


def training_step(model, batch, loss_func, miner_func, device):
    images, labels = batch
    images = images.to(device)
    embeddings = model(images)
    miner_output = miner_func(embeddings, labels)

    return loss_func(embeddings, labels, miner_output)


def train_model(model, loss_func, miner_func, train_loader, test_loader, train_eval_ds, test_eval_ds, epochs, device,
                weight_dir, name):

    os.makedirs(weight_dir, exist_ok=True)
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    best_acc = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    acc_calc = AccuracyCalculator(include=("precision_at_1",), k=1)

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch in tqdm(train_loader):
            loss = training_step(model, batch, loss_func, miner_func, device)

            # update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.cpu().detach().numpy())

        train_loss.append(np.mean(batch_losses))
        train_acc.append(evaluate(model, train_eval_ds, device, acc_calc))

        model.eval()
        batch_losses = []
        for batch in test_loader:
            loss = training_step(model, batch, loss_func, miner_func, device)

            batch_losses.append(loss.cpu().detach().numpy())

        test_loss.append(np.mean(batch_losses))
        test_acc.append(evaluate(model, test_eval_ds, device, acc_calc))

        epoch_end(epoch, train_loss[-1], test_loss[-1], train_acc[-1], test_acc[-1])

        # save model
        if best_acc < (train_acc[-1] + test_acc[-1]) / 2:
            best_acc = (train_acc[-1] + test_acc[-1]) / 2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(weight_dir, f'{name}.pt'))

    return train_loss, train_acc, test_loss, test_acc


def get_miner_and_loss_func(name, data_set, model):
    if name == 'npair':
        loss_func = losses.NPairsLoss()
        miner_func = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.2)

    if name == 'triplet':
        loss_func = losses.TripletMarginLoss(margin=0.05,
                                             swap=False,
                                             smooth_loss=False,
                                             triplets_per_anchor="all")
        miner_func = miners.TripletMarginMiner(margin=10, type_of_triplets="all")

    if name == 'arcface':
        loss_func = losses.ArcFaceLoss(len(data_set.data.label.unique()),
                                       list(model.children())[-1].out_features,
                                       margin=4,
                                       scale=1)
        miner_func = miners.AngularMiner(angle=3)

    if name == 'sphereface':
        loss_func = losses.SphereFaceLoss(len(data_set.data.label.unique()),
                                          list(model.children())[-1].out_features,
                                          margin=4,
                                          scale=1)
        miner_func = miners.AngularMiner(angle=3)
    return miner_func, loss_func


def save_history(save_dir, train_loss, train_acc, test_loss, test_acc):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'train_loss.pt'), 'wb') as file:
        pickle.dump(train_loss, file)
    with open(os.path.join(save_dir, 'train_acc.pt'), 'wb') as file:
        pickle.dump(train_acc, file)
    with open(os.path.join(save_dir, 'test_loss.pt'), 'wb') as file:
        pickle.dump(test_loss, file)
    with open(os.path.join(save_dir, 'test_acc.pt'), 'wb') as file:
        pickle.dump(test_acc, file)
    print(f'History saved to directory: {save_dir}')


def get_auto_encoder_model():
    encoder = nn.Sequential(
        nn.Conv2d(3, 32, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4),
        nn.Conv2d(64, 128, 4),
        nn.Linear(215, 128),
        nn.Dropout(.2),
        nn.Linear(128, 128)
    )
    decoder = nn.Sequential(
        nn.Linear(128, 128),
        nn.Dropout(.1),
        nn.Linear(128, 215),
        nn.Conv2d(128, 64, 4, 1, 3),
        nn.Conv2d(64, 32, 4, 1, 3),
        nn.ReLU(),
        nn.Conv2d(32, 3, 4, 1, 3),
    )
    auto_encoder = nn.Sequential(encoder, decoder)
    return auto_encoder


def train_autoencoder_model(model, train_loader, test_loader, epochs, weight_dir, name, device):
    os.makedirs(weight_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_func = nn.MSELoss()

    best_loss = None
    train_history, test_history = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for batch in tqdm(train_loader):
            images, labels = batch
            images = images.to(device)
            optimizer.zero_grad()
            restored_images = model(images)
            loss = loss_func(images, restored_images)
            del images, restored_images
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().detach().numpy())

        train_history.append(np.mean(epoch_loss))

        model.eval()
        epoch_loss = []
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            restored_images = model(images)
            loss = loss_func(images, restored_images)
            del images, restored_images
            epoch_loss.append(loss.cpu().detach().numpy())

        test_history.append(np.mean(epoch_loss))

        # save model
       # if best_loss is None or best_loss > (train_history[-1] + test_history[-1]) / 2:
        #best_loss = (train_history[-1] + test_history[-1]) / 2
        #torch.save({
        #    'epoch': epoch,
        #    'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'loss': loss,
        #    }, os.path.join(weight_dir, f'{name}.pt'))
        #    #}, os.path.join(weight_dir, 'auto.pt'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(weight_dir, f'{name}.pt'))

        print(f'Epoch {epoch}:')
        print(f'train loss: {train_history[-1]}')
        print(f'test loss: {test_history[-1]}')

    return train_history, test_history


def save_history_autoencoder(save_dir, train_loss, test_loss):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'train_loss.pt'), 'wb') as file:
        pickle.dump(train_loss, file)
    with open(os.path.join(save_dir, 'test_loss.pt'), 'wb') as file:
        pickle.dump(test_loss, file)
    print(f'History saved to directory: {save_dir}')
