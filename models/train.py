import os
from collections import OrderedDict
import pickle
from sklearn.decomposition import PCA
from pytorch_metric_learning import testers

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from utils.training import (
    get_miner_and_loss_func,
    train_model,
    save_history,
    save_history_autoencoder,
    train_autoencoder_model,
    get_auto_encoder_model,
)

from utils.visualization import save_history_acc_graph, save_history_loss_graph

from utils.datasets import (
    VGG_FACE_DS,
    VGG_FACE_ADVERSARIAL,
    VGG_FACE_ADVERSARIAL_AUTOENCODER,
    get_support_query_datasets,
    get_origin_support_and_adversarial_query
)

from models.dir_settings import ROOT_DIR, DATA_BASE_DIR, ADV_DIR


def train_origin_model(loss_name, num_epochs, batch_size=32, query_size=0.75, model_name='origin'):
    train_path = os.path.join(DATA_BASE_DIR, 'train')
    test_path = os.path.join(DATA_BASE_DIR, 'test')
    model_weight_dir = os.path.join(ROOT_DIR, f'weights/{model_name}')
    history_save_dir = os.path.join(ROOT_DIR, f'history/{model_name}/{loss_name}')
    graph_save_dir = os.path.join(ROOT_DIR, f'graphs/{model_name}/{loss_name}')

    # GET DATA
    ##################################################################
    train_data_set = VGG_FACE_DS(train_path)
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
    )
    train_eval_ds = get_support_query_datasets(train_data_set, query_size=query_size)
    test_data_set = VGG_FACE_DS(test_path)
    test_data_loader = DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=True,
    )
    test_eval_ds = get_support_query_datasets(train_data_set, query_size=query_size)
    ##################################################################

    # TRAINING
    ##################################################################
    model = resnet50(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    miner_func, loss_func = get_miner_and_loss_func(loss_name, train_data_set, model)

    train_loss, train_acc, test_loss, test_acc = train_model(model,
                                                             loss_func,
                                                             miner_func,
                                                             train_data_loader,
                                                             test_data_loader,
                                                             train_eval_ds,
                                                             test_eval_ds,
                                                             num_epochs,
                                                             device,
                                                             model_weight_dir,
                                                             loss_name)

    save_history(history_save_dir, train_loss, train_acc, test_loss, test_acc)
    save_history_loss_graph(graph_save_dir, train_loss, test_loss, loss_name)
    save_history_acc_graph(graph_save_dir, train_acc, test_acc, loss_name)
    ##################################################################


def train_model_with_adversarial(attack_name, loss_name, num_epochs, batch_size=32, query_size=0.75,
                                 model_name='resnet_with_adversarial'):
    train_origin_path = os.path.join(DATA_BASE_DIR, 'train')
    train_adversarial_path = os.path.join(ADV_DIR, f'train/{attack_name}')
    test_origin_path = os.path.join(DATA_BASE_DIR, 'test')
    test_adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    model_weight_dir = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}')
    history_save_dir = os.path.join(ROOT_DIR, f'history/adversarial/{model_name}/{attack_name}/{loss_name}')
    graph_save_dir = os.path.join(ROOT_DIR, f'graphs/adversarial/{model_name}/{attack_name}/{loss_name}')

    # GET DATA
    ##################################################################
    train_data_set = VGG_FACE_ADVERSARIAL(
        train_origin_path,
        train_adversarial_path
    )
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data_set = VGG_FACE_ADVERSARIAL(
        test_origin_path,
        test_adversarial_path
    )
    test_data_loader = DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=True,
    )
    train_eval_ds = get_origin_support_and_adversarial_query(train_adversarial_path, attack_name, train_origin_path,
                                                             query_size=query_size, state=42)
    test_eval_ds = get_origin_support_and_adversarial_query(test_adversarial_path, attack_name, test_origin_path,
                                                            query_size=query_size, state=42)
    ##################################################################

    # TRAIN
    ##################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=True)
    model.to(device)

    miner_func, loss_func = get_miner_and_loss_func(loss_name, train_data_set, model)

    train_loss, train_acc, test_loss, test_acc = train_model(model,
                                                             loss_func,
                                                             miner_func,
                                                             train_data_loader,
                                                             test_data_loader,
                                                             train_eval_ds,
                                                             test_eval_ds,
                                                             num_epochs,
                                                             device,
                                                             model_weight_dir,
                                                             loss_name)

    save_history(history_save_dir, train_loss, train_acc, test_loss, test_acc)
    save_history_loss_graph(graph_save_dir, train_loss, test_loss, loss_name)
    save_history_acc_graph(graph_save_dir, train_acc, test_acc, loss_name)
    ##################################################################


def train_autoencoder(attack_name, num_epochs, batch_size=32, model_name='autoencoder'):
    train_origin_path = os.path.join(DATA_BASE_DIR, 'train')
    train_adversarial_path = os.path.join(ADV_DIR, f'train/{attack_name}')
    test_origin_path = os.path.join(DATA_BASE_DIR, 'test')
    test_adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    model_weight_dir = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}')
    history_save_dir = os.path.join(ROOT_DIR, f'history/adversarial/{model_name}/{attack_name}')
    graph_save_dir = os.path.join(ROOT_DIR, f'graphs/adversarial/{model_name}/{attack_name}')

    auto_encoder = get_auto_encoder_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_encoder.to(device)

    # GET DATA
    ##################################################################
 
    train_data_set = VGG_FACE_ADVERSARIAL_AUTOENCODER(
    train_origin_path,
    train_adversarial_path,
    )
    train_loader = DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True
    )
    print('OK train datasets')
    test_data_set = VGG_FACE_ADVERSARIAL_AUTOENCODER(
    test_origin_path,
    test_adversarial_path,
    )

    test_loader = DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=True
    )
    print('OK test datasets')
    ##################################################################

    # TRAIN
    ##################################################################
    loss_name = 'mse'
    train_loss, test_loss = train_autoencoder_model(auto_encoder, train_loader, test_loader, num_epochs,
                                                    model_weight_dir, loss_name, device)
    save_history_autoencoder(history_save_dir, train_loss, test_loss)
    save_history_loss_graph(graph_save_dir, train_loss, test_loss, loss_name)
    ##################################################################


def train_autoencoder_and_model(attack_name, loss_name, num_epochs, batch_size=32, query_size=0.75,
                                model_name='autoencoder_and_model'):
    train_origin_path = os.path.join(DATA_BASE_DIR, 'train')
    train_adversarial_path = os.path.join(ADV_DIR, f'train/{attack_name}')
    test_origin_path = os.path.join(DATA_BASE_DIR, 'test')
    test_adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    model_weight_dir = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}')
    history_save_dir = os.path.join(ROOT_DIR, f'history/adversarial/{model_name}/{attack_name}/{loss_name}')
    graph_save_dir = os.path.join(ROOT_DIR, f'graphs/adversarial/{model_name}/{attack_name}/{loss_name}')

    # GET DATA
    ##################################################################
    train_data_set = VGG_FACE_ADVERSARIAL(train_origin_path, train_adversarial_path)
    train_loader = DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data_set = VGG_FACE_ADVERSARIAL(test_origin_path, test_adversarial_path)
    test_loader = DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=True,
    )

    train_eval_ds = get_origin_support_and_adversarial_query(train_adversarial_path, attack_name, train_origin_path,
                                                             query_size=query_size, state=42)
    test_eval_ds = get_origin_support_and_adversarial_query(test_adversarial_path, attack_name, test_origin_path,
                                                            query_size=query_size, state=42)
    ##################################################################

    # TRAIN
    ##################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = get_auto_encoder_model()
    resnet = resnet50(pretrained=True)
    model = nn.Sequential(
        autoencoder,
        resnet
    )
    model.to(device)

    miner_func, loss_func = get_miner_and_loss_func(loss_name, train_data_set, model)

    train_loss, train_acc, test_loss, test_acc = train_model(model,
                                                             loss_func,
                                                             miner_func,
                                                             train_loader,
                                                             test_loader,
                                                             train_eval_ds,
                                                             test_eval_ds,
                                                             num_epochs,
                                                             device,
                                                             model_weight_dir,
                                                             loss_name)

    save_history(history_save_dir, train_loss, train_acc, test_loss, test_acc)
    save_history_loss_graph(graph_save_dir, train_loss, test_loss, loss_name)
    save_history_acc_graph(graph_save_dir, train_acc, test_acc, loss_name)
    ##################################################################


def training_pca(loss_name, n_components, attack_name, batch_size, model_name='pca'):
    train_origin_path = os.path.join(DATA_BASE_DIR, 'train')
    train_adversarial_path = os.path.join(ADV_DIR, f'train/{attack_name}')
    model_weight_dir = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}/{n_components}')
    resnet_weight_path = os.path.join(ROOT_DIR, f'weights/origin/{loss_name}.pt')

    train_data_set = VGG_FACE_ADVERSARIAL(train_origin_path, train_adversarial_path)

    dnn_model = resnet50()
    dnn_model.load_state_dict(torch.load(resnet_weight_path, map_location=torch.device('cpu'))['model_state_dict'])

    pca_model = PCA(n_components=n_components)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tester = testers.BaseTester(data_device=device, batch_size=batch_size, dataloader_num_workers=0)
    embeddings, labels = tester.get_all_embeddings(train_data_set, dnn_model)
    pca_model.fit(embeddings.cpu())

    # save models weights
    os.makedirs(model_weight_dir, exist_ok=True)
    model_weight_path = os.path.join(model_weight_dir, 'pca.pt')
    pickle.dump(pca_model, open(model_weight_path, 'wb'))


def train_simple_model_after_resnet(attack_name, loss_name, num_epochs, batch_size=32,
                                    query_size=0.75, model_name='simple_model_after_resnet'):
    train_origin_path = os.path.join(DATA_BASE_DIR, 'train')
    train_adversarial_path = os.path.join(ADV_DIR, f'train/{attack_name}')
    test_origin_path = os.path.join(DATA_BASE_DIR, 'test')
    test_adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    resnet_weight_path = os.path.join(ROOT_DIR, f'weights/origin/{loss_name}.pt')
    model_weight_dir = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}')
    history_save_dir = os.path.join(ROOT_DIR, f'history/adversarial/{model_name}/{attack_name}')
    graph_save_dir = os.path.join(ROOT_DIR, f'graphs/adversarial/{model_name}/{attack_name}/{loss_name}')

    # BUILD MODEL
    ##################################################################
    res_model = resnet50()
    res_model.load_state_dict(torch.load(resnet_weight_path, map_location=torch.device('cpu'))['model_state_dict'])

    second_model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.Dropout(0.1),
        nn.Tanh(),
        nn.Linear(500, 500)
    )

    model = nn.Sequential(OrderedDict([
        ('resnet50', res_model),
        ('second_model', second_model)
    ]))

    for param in model.resnet50.parameters():
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ##################################################################

    # GET DATA
    ##################################################################
    train_dataset = VGG_FACE_ADVERSARIAL(train_origin_path, train_adversarial_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data_set = VGG_FACE_ADVERSARIAL(test_origin_path, test_adversarial_path)
    test_loader = DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=True,
    )

    train_eval_ds = get_origin_support_and_adversarial_query(train_adversarial_path, attack_name, train_origin_path,
                                                             query_size=query_size, state=42)
    test_eval_ds = get_origin_support_and_adversarial_query(test_adversarial_path, attack_name, test_origin_path,
                                                            query_size=query_size, state=42)
    ##################################################################

    # TRAIN
    ##################################################################
    miner_func, loss_func = get_miner_and_loss_func(loss_name, train_dataset, model)
    train_loss, train_acc, test_loss, test_acc = train_model(model,
                                                             loss_func,
                                                             miner_func,
                                                             train_loader,
                                                             test_loader,
                                                             train_eval_ds,
                                                             test_eval_ds,
                                                             num_epochs,
                                                             device,
                                                             model_weight_dir,
                                                             loss_name)

    save_history(history_save_dir, train_loss, train_acc, test_loss, test_acc)
    save_history_loss_graph(graph_save_dir, train_loss, test_loss, loss_name)
    save_history_acc_graph(graph_save_dir, train_acc, test_acc, loss_name)
    ##################################################################
