import os
import pickle
import plotly.graph_objects as go
import numpy as np


def save_history_loss_graph(save_dir, train_loss, test_loss, loss_name):
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(0, len(train_loss), 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=train_loss, name='train'))
    fig.add_trace(go.Scatter(x=x, y=test_loss, name='test'))
    fig.update_layout(
        title_text=f'Функция потерь {loss_name.upper()}',
        title_x=0.5,
        yaxis_title='Значения функции потерь',
        xaxis_title='Эпохи',
        legend=dict(y=.5)
    )
    fig.write_image(os.path.join(save_dir, f'loss_{loss_name}.png'))


def save_history_acc_graph(save_dir, train_acc, test_acc, name):
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(0, len(train_acc), 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=train_acc, name='train'))
    fig.add_trace(go.Scatter(x=x, y=test_acc, name='test'))
    fig.update_layout(
        title_text='Показатель качества модели',
        title_x=0.5,
        yaxis_title='Значения показателя качества',
        xaxis_title='Эпохи',
        legend=dict(y=.5)
    )
    fig.write_image(os.path.join(save_dir, f'accuracy_{name}.png'))


def plot_graph(history_path, name, save_dir):
    with open(f'{history_path}/train_acc.pt', 'rb') as file:
        train_acc = pickle.load(file)
    with open(f'{history_path}/test_acc.pt', 'rb') as file:
        test_acc = pickle.load(file)
    with open(f'{history_path}/train_loss.pt', 'rb') as file:
        train_loss = pickle.load(file)
    with open(f'{history_path}/test_loss.pt', 'rb') as file:
        test_loss = pickle.load(file)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    x_val = np.arange(0, len(train_loss), 1)

    # loss function
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_val, y=train_loss, name='train'))
    fig.add_trace(go.Scatter(x=x_val, y=test_loss, name='test'))
    fig.update_layout(
        title_text='N-Pair Loss Function',
        title_x=0.5,
        yaxis_title='Loss values',
        xaxis_title='epochs',
        legend=dict(y=.5)
    )
    fig.write_image(os.path.join(save_dir, f'loss_{name}.png'))

    # acc metrics
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_val, y=train_acc, name='train'))
    fig.add_trace(go.Scatter(x=x_val, y=test_acc, name='test'))
    fig.update_layout(
        title_text='Model accuracy',
        title_x=0.5,
        yaxis_title='Accuracy values',
        xaxis_title='epochs',
        legend=dict(y=.5)
    )
    fig.write_image(os.path.join(save_dir, f'accuracy_{name}.png'))
