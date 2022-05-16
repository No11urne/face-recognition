import os
import pickle
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models import resnet50

from utils.training import get_auto_encoder_model
from utils.testing import eval_model, eval_with_pca

from models.dir_settings import ROOT_DIR, DATA_BASE_DIR, ADV_DIR


def eval_resnet50(loss_name, attack_name, batch_size=32):
    origin_path = os.path.join(DATA_BASE_DIR, 'test')
    adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    model_weights_path = os.path.join(ROOT_DIR, f'weights/origin/{loss_name}.pt')

    model = resnet50()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'))['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score = eval_model(origin_path, model, device, batch_size, False)
    print(f'VGG_FACE2 ({loss_name}, {attack_name}) Origin resnet50 accuracy is: {score}')

    score = eval_model(origin_path, model, device, batch_size, True, adversarial_path)
    print(f'ADVERS ({loss_name}, {attack_name}) Origin resnet50 accuracy is: {score}')


def eval_resnet50_with_adversarial(loss_name, attack_name, batch_size=32, model_name='resnet_with_adversarial'):
    origin_path = os.path.join(DATA_BASE_DIR, 'test')
    adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    model_weight_path = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}/{loss_name}.pt')

    model = resnet50()
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu'))['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score = eval_model(origin_path, model, device, batch_size, False)
    print(f'VGG_FACE2 ({loss_name}, {attack_name}) Resnet50 with adversarial images accuracy is: {score}')

    score = eval_model(origin_path, model, device, batch_size, True, adversarial_path)
    print(f'ADVERS ({loss_name}, {attack_name}) Resnet50 with adversarial images accuracy is: {score}')


def eval_autoencoder(loss_name, attack_name, batch_size=32, model_name='autoencoder'):
    origin_path = os.path.join(DATA_BASE_DIR, 'test')
    adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    autoencoder_weight_path = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}/mse.pt')
    resnet50_model_weight_path = os.path.join(ROOT_DIR, f'weights/origin/{loss_name}.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = get_auto_encoder_model()
    autoencoder.load_state_dict(
        torch.load(autoencoder_weight_path, map_location=torch.device('cpu'))['model_state_dict'])

    model = resnet50()
    model.load_state_dict(torch.load(resnet50_model_weight_path, map_location=torch.device('cpu'))['model_state_dict'])

    model = nn.Sequential(
        autoencoder,
        model
    )
    score = eval_model(origin_path, model, device, batch_size, False)
    print(f'VGG_FACE2 ({loss_name}, {attack_name}) Autoencoder with fitted resnet50 model accuracy is: {score}')
    score = eval_model(origin_path, model, device, batch_size, True, adversarial_path)
    print(f'ADVERS ({loss_name}, {attack_name}) Autoencoder with fitted resnet50 model on accuracy is: {score}')


def eval_autoencoder_resnet50(loss_name, attack_name, batch_size=32, model_name='autoencoder_and_model'):
    origin_path = os.path.join(DATA_BASE_DIR, 'test')
    adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    model_weight_path = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}/{loss_name}.pt')

    model = nn.Sequential(
        get_auto_encoder_model(),
        resnet50()
    )
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu'))['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = eval_model(origin_path, model, device, batch_size, False)
    print(f'VGG_FACE2 ({loss_name}, {attack_name}) Autoencoder and CONST resnet50 accuracy is: {score}')

    score = eval_model(origin_path, model, device, batch_size, True, adversarial_path)
    print(f'ADVERS ({loss_name}, {attack_name}) Autoencoder and CONST resnet50 accuracy is: {score}')


def eval_pca(loss_name, attack_name, n_components, batch_size=32, model_name='pca'):
    origin_path = os.path.join(DATA_BASE_DIR, 'test')
    adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    resnet_weight_path = os.path.join(ROOT_DIR, f'weights/origin/{loss_name}.pt')
    pca_model_weight_path = os.path.join(ROOT_DIR,
                                         f'weights/adversarial/{model_name}/{attack_name}/{n_components}/pca.pt')

    dnn_model = resnet50()
    dnn_model.load_state_dict(torch.load(resnet_weight_path, map_location=torch.device('cpu'))['model_state_dict'])

    pca_model = pickle.load(open(pca_model_weight_path, 'rb'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = eval_with_pca(dnn_model, pca_model, origin_path, device, batch_size, False)
    print(f'VGG_FACE2 ({loss_name}, {attack_name}) CONST resnet50 and PCA accuracy is: {score}')

    score = eval_with_pca(dnn_model, pca_model, origin_path, device, batch_size, True, adversarial_path, )
    print(f'ADVERS ({loss_name}, {attack_name}) CONST resnet50 and PCA accuracy is: {score}')


def eval_simple_model_after_resnet50(loss_name, attack_name, batch_size=32, model_name='simple_model_after_resnet'):
    origin_path = os.path.join(DATA_BASE_DIR, 'test')
    adversarial_path = os.path.join(ADV_DIR, f'test/{attack_name}')
    model_weight_path = os.path.join(ROOT_DIR, f'weights/adversarial/{model_name}/{attack_name}/{loss_name}.pt')

    resnet_model = resnet50()
    simple_model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.Dropout(0.1),
        nn.Tanh(),
        nn.Linear(500, 500)
    )
    model = nn.Sequential(OrderedDict([
        ('resnet50', resnet_model),
        ('second_model', simple_model)
    ]))
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu'))['model_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = eval_model(origin_path, model, device, batch_size, False)
    print(
        f'VGG_FACE2 ({loss_name}, {attack_name}) CONST resnet50 and simple model ({loss_name}, {attack_name}) accuracy is: {score}')

    score = eval_model(origin_path, model, device, batch_size, True, adversarial_path)
    print(
        f'ADVERS ({loss_name}, {attack_name}) CONST resnet50 and simple model ({loss_name}, {attack_name}) accuracy is: {score}')
