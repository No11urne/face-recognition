import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from facenet_pytorch import MTCNN

from models.dir_settings import ADV_DIR, DATA_BASE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(select_largest=True, post_process=True, device=device)


def read_data(root_path):
    data, n_iter = pd.DataFrame(columns=['image_path', 'label']), 0
    label_list = os.listdir(root_path)[:50]
    if '.DS_Store' in label_list: label_list.remove('.DS_Store')
    le = LabelEncoder()
    le.fit(label_list)
    for dir_path in label_list:
        list_images = os.listdir(os.path.join(root_path, dir_path))
        if '.DS_Store' in list_images: list_images.remove('.DS_Store')
        for image in list_images:
            data.loc[n_iter] = {
                'image_path': os.path.join(root_path, os.path.join(dir_path, image)),
                'label': le.transform([dir_path])[0]
            }
            n_iter += 1
    return data, le


def image_transform(image):
    face_detection = transforms.Compose([
        mtcnn,
        transforms.ToPILImage()
    ])
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        image = face_detection(image)
        image = preprocess(image)
    except:
        image = preprocess(image)
    return image


class VGG_FACE_DS(Dataset):
    # Для обучения модели resnet50
    def __init__(self, root_path):
        super(VGG_FACE_DS, self).__init__()
        self.data, self.l_enc = read_data(root_path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_path, label = self.data.iloc[idx]
        with Image.open(img_path) as image:
            image = image_transform(image)
        return image, label


class Eval_DS(Dataset):
    # Для оценки показателя качества модели resnet50
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_path, label = self.data.iloc[idx]
        with Image.open(img_path) as image:
            image = image_transform(image)
        return image, label


class AdversarialDS(Dataset):
    # Для формирования враждебных изображения
    def __init__(self, root_path, num_classes):
        super(AdversarialDS, self).__init__()
        self.data, self.l_enc = read_data(root_path)
        classes = self.data.label.unique()[:num_classes]
        self.data = self.data[self.data.label.isin(classes)]

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_path, label = self.data.iloc[idx]
        img_name = img_path[img_path.rfind('/') + 1:]
        with Image.open(img_path) as image:
            image = self.preprocess(image)
        return image.unsqueeze(0), label, img_name


class VGG_FACE_ADVERSARIAL(Dataset):
    # Для обучения и оценки показателя качесвта модели с "защитой"
    def __init__(self, origin_path, adversarial_path):
        super().__init__()
        orig_data, self.l_enc = read_data(origin_path)
        adversarial_data, _ = read_data(adversarial_path)
        orig_data = orig_data[orig_data.label.isin(adversarial_data.label.unique())]

        self.data = pd.concat([
            orig_data,
            adversarial_data
        ]).sample(frac=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_path, label = self.data.iloc[idx]
        with Image.open(img_path) as image:
            image = image_transform(image)
        return image, label


class VGG_FACE_ADVERSARIAL_AUTOENCODER(Dataset):
    # Для обучения автоэнкодера: x - origin_img or advers_img; y - origin_img
    def __init__(self, origin_path, adversarial_path):
        origin_data, self.l_enc = read_data(origin_path)
        adversarial_data, _ = read_data(adversarial_path)
        adversarial_data.label = adversarial_data.image_path.apply(
            lambda x: x.replace(f'{adversarial_path}', origin_path))
        origin_data.label = origin_data.image_path
        self.data = pd.concat([
            origin_data,
            adversarial_data
        ]).sample(frac=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        x_img_path, y_img_path = self.data.iloc[item]
        with Image.open(x_img_path) as image:
            x_image = image_transform(image)
        with Image.open(y_img_path) as image:
            y_image = image_transform(image)
        return x_image, y_image


def get_support_query_datasets(data_set, query_size=0.75, state=42):
    support_df, query_df = train_test_split(data_set.data, test_size=query_size, random_state=state)
    support_data_set, query_data_set = Eval_DS(support_df), Eval_DS(query_df)
    return support_data_set, query_data_set


def get_origin_support_and_adversarial_query(adversarial_dir, attack_name, origin_dir, query_size=0.75, state=42):
    data, l_enc = read_data(adversarial_dir)
    support_df, query_df = train_test_split(data, test_size=query_size, random_state=state)

    support_df = support_df.apply(
        lambda path: path.replace(
            os.path.join(f'{ADV_DIR}/test', attack_name),
            f'{DATA_BASE_DIR}/test')
    )

    support_data_set, query_data_set = Eval_DS(support_df), Eval_DS(query_df)
    return support_data_set, query_data_set
