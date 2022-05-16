import torch
from sklearn.model_selection import train_test_split
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from utils.datasets import VGG_FACE_DS, Eval_DS


def evaluate(model, data_set, device, acc_calc, batch_size=32):
    model.to(device)
    model.eval()
    tester = testers.BaseTester(data_device=device, batch_size=batch_size, dataloader_num_workers=0)
    support_dataset, query_dataset = data_set
    support_embeddings, support_labels = tester.get_all_embeddings(support_dataset, model)
    support_labels = support_labels.squeeze(1)

    query_embeddings, query_labels = tester.get_all_embeddings(query_dataset, model)
    query_labels = query_labels.squeeze(1)

    accuracies = acc_calc.get_accuracy(support_embeddings,
                                       query_embeddings,
                                       support_labels,
                                       query_labels,
                                       False)
    return accuracies['precision_at_1']


def eval_model(test_origin_path, model, device, batch_size=32, is_adversarial=False, adversarial_path=None):
    if not is_adversarial:
        attack_ds = VGG_FACE_DS(test_origin_path)
    else:
        attack_ds = VGG_FACE_DS(adversarial_path)
    support_df, query_df = train_test_split(attack_ds.data, test_size=0.75, random_state=42)
    if is_adversarial:
        support_df.image_path = support_df.image_path.apply(lambda x: x.replace(adversarial_path, test_origin_path))

    support_dataset = Eval_DS(support_df)
    query_dataset = Eval_DS(query_df)

    acc_calc = AccuracyCalculator(include=("precision_at_1",), k=1)

    return evaluate(model, (support_dataset, query_dataset), device, acc_calc, batch_size)


def eval_with_pca(dnn_model, pca_model, test_origin_path, device, batch_size=32, is_adversarial=False,
                  adversarial_path=None):
    dnn_model.to(device)
    dnn_model.eval()
    tester = testers.BaseTester(data_device=device, batch_size=batch_size, dataloader_num_workers=0)

    attack_ds = VGG_FACE_DS(test_origin_path)
    support_df, query_df = train_test_split(attack_ds.data, test_size=0.75, random_state=42)
    if is_adversarial:
        support_df.image_path = support_df.image_path.apply(lambda x: x.replace(test_origin_path, adversarial_path))

    support_dataset, query_dataset = Eval_DS(support_df), Eval_DS(query_df)

    support_embeddings, support_labels = tester.get_all_embeddings(support_dataset, dnn_model)
    support_labels = support_labels.squeeze(1)
    support_embeddings = pca_model.transform(support_embeddings.cpu())
    support_embeddings = torch.tensor(support_embeddings).to(device)

    query_embeddings, query_labels = tester.get_all_embeddings(query_dataset, dnn_model)
    query_labels = query_labels.squeeze(1)
    query_embeddings = pca_model.transform(query_embeddings.cpu())
    query_embeddings = torch.tensor(query_embeddings).to(device)

    acc_calc = AccuracyCalculator(include=("precision_at_1",), k=1)
    accuracies = acc_calc.get_accuracy(support_embeddings,
                                       query_embeddings,
                                       support_labels,
                                       query_labels,
                                       False)
    return accuracies['precision_at_1']
