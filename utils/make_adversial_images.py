import os
from tqdm import tqdm

import torch
from torchvision.models import resnet50
from torchvision.utils import save_image
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, CarliniL2Method

from utils.datasets import AdversarialDS


def get_model(device, model_weights_path):
    model = resnet50(pretrained=True).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'))['model_state_dict'])
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), .1, momentum=0.9, weight_decay=1e-4)
    wrapped_model = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        clip_values=(0, 1),
        input_shape=(3, 244, 244),
        nb_classes=1000,
        preprocessing=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )
    return wrapped_model


def get_attack(name, wrapped_model):
    if name == 'pgd':
        return ProjectedGradientDescent(wrapped_model, norm=2)
    elif name == 'deepfool':
        return DeepFool(wrapped_model, max_iter=100)
    elif name == 'cw':
        return CarliniL2Method(wrapped_model, max_iter=100)
    elif name == 'fgm':
        return FastGradientMethod(wrapped_model, norm=2)
    else:
        print('Unseen attack type')
        return None


def save_attack_images(root_dir, save_dir, attack_name, device, model_weights_path, num_classes):
    if not os.path.isfile(model_weights_path):
        print('Incorrect model weight path')
        return None

    wrapped_model = get_model(device, model_weights_path)
    attack = get_attack(attack_name, wrapped_model)
    attack_set = AdversarialDS(root_dir, num_classes)

    for el in tqdm(attack_set):
        # read img
        image, label, img_name = el
        dir_name = attack_set.l_enc.inverse_transform([label])[0]

        # make attack
        adv_images_pgd = attack.generate(x=image.numpy(), verbose=True)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        attack_dir = os.path.join(save_dir, attack_name)
        if not os.path.isdir(attack_dir):
            os.mkdir(attack_dir)

        class_dir = os.path.join(attack_dir, dir_name)
        if not os.path.isdir(class_dir):
            os.mkdir(class_dir)

        # save image
        save_image(torch.tensor(adv_images_pgd), os.path.join(class_dir, img_name))

    print('Attack is end!')


if __name__ == '__main__':
    root_dir = '/content/drive/MyDrive/VGG_FACE/vgg_data/test'
    save_dir = '/content/drive/MyDrive/VGG_FACE/vgg_data/adversial_images/test'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights_path = '/content/drive/MyDrive/VGG_FACE/weights/npair.pt'
    num_classes = 1
    save_attack_images(root_dir, save_dir, 'pgd', device, model_weights_path, num_classes)
    # save_attack_images(root_dir, save_dir, 'deepfool', device, model_weights_path, num_classes)
    # save_attack_images(root_dir, save_dir, 'cw', device, model_weights_path, num_classes)
    # save_attack_images(root_dir, save_dir, 'fgm', device, model_weights_path, num_classes)
