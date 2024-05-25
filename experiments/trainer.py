import os
from logging import Logger

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.models.base_model import BaseModel


class Trainer:
    def __init__(self, logger: Logger, model: BaseModel, device, early_stop=None):
        self.logger = logger
        self.device = device
        self.model = model
        self.early_stop = early_stop

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        data_loaders = {
            'train': train_loader,
            'val': val_loader
        }
        optimizer = self.model.optimizer()
        criterion = self.model.criterion()
        loss_history = {'train': [], 'val': []}
        metric_history = {'train': [], 'val': []}
        best_metric = 0.0
        best_epoch = 0
        stop_training = False

        for epoch in range(num_epochs):
            if stop_training:
                break

            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_metric = 0

                for inputs, labels in tqdm(data_loaders[phase]):
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        preds = self.model.predictions(outputs)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * preds.size(0)
                    running_metric += self.model.metric(labels, preds)

                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                epoch_metric = running_metric / len(data_loaders[phase].dataset)

                self.logger.info(f'{phase} Loss: {epoch_loss:.4f} Metric: {epoch_metric:.4f}')

                if phase == 'val' and epoch_metric > best_metric:
                    best_metric = epoch_metric
                    best_epoch = epoch
                    self.model.save_checkpoint(epoch)

                loss_history[phase].append(epoch_loss)
                metric_history[phase].append(epoch_metric)
                self.update_graphs(loss_history, metric_history)

                if self.early_stop and epoch - best_epoch >= self.early_stop:
                    self.logger.info(f'Validation set did not improve for {self.early_stop} epochs, stop training.')
                    stop_training = True
                    break

        self.logger.info(f'Training completed. Best epoch={best_epoch}, best metric={best_metric}')

    def update_graphs(self, loss_history, metric_history):
        self.draw_graph(loss_history, 'loss')
        self.draw_graph(metric_history, 'metric')

    def draw_graph(self, values: dict, title):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title)
        for values_type in values:
            epochs = list(range(len(values[values_type])))
            plot_values = values[values_type]
            axs.plot(epochs, plot_values, label=values_type)

        axs.legend()
        graph_dir = os.path.join(self.model.output_dir, 'train_graphs')
        os.makedirs(graph_dir, exist_ok=True)
        fig.savefig(os.path.join(graph_dir, f'{title}.png'))
        plt.close(fig)


