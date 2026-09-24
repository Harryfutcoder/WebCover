import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src import settings


class DomElemDiscriminator(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(DomElemDiscriminator, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        return out


class DataCache:
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.ad_dict = {}  # html -> 0 / 1

    # def clear(self):
    def update_by_explore(self, data, label, embedding):
        # ad_dict
        update_index = []
        for index in range(len(data)):
            if data[index] in self.ad_dict:
                pass  # label conflict: keep existing label, skip update
            else:
                self.ad_dict[data[index]] = label[index]
                update_index.append(index)
        # get update data
        if len(update_index):
            update_index = np.array(update_index)
            update_embedding = embedding.copy()[update_index]
            update_label = np.array(label)[update_index]
            # update_label = np.array([update_label])
            # train data and label
            if self.train_data is not None:
                self.train_data = np.concatenate((self.train_data, update_embedding), axis=0)
            else:
                self.train_data = update_embedding
            if self.train_label is not None:
                self.train_label = np.concatenate((self.train_label, update_label))
            else:
                self.train_label = update_label


class ActionDiscriminator:
    def __init__(self):
        # config
        self.flag_on = True
        self.epochs_per_train = 25
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.embedding_model = SentenceTransformer(settings.BERT_MODEL_PATH).to(self.device)
        self.discriminator = DomElemDiscriminator(384, 256, 2).to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.01)
        self.loss_function = nn.CrossEntropyLoss()

        self.data_cache = DataCache()
        self.ad_cache = {}
        # record
        self.flag_enable = False
        self.flag_explore = False
        self.time_cost = 0

    def _recognize_by_discriminator(self, elements):
        embedding = self.embedding_model.encode(elements)
        embedding = torch.tensor(embedding).to(self.device)
        self.discriminator.eval()
        with torch.no_grad():
            outputs = self.discriminator(embedding)
            outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        return outputs

    def recognize(self, potential_actions):
        start_time = time.time()
        action_space = []
        elements_index, elements = [], []
        for index in range(len(potential_actions)):
            action_info = potential_actions[index]
            # data cache
            if action_info['outerHTML'] in self.data_cache.ad_dict:
                if self.data_cache.ad_dict[action_info['outerHTML']]:
                    action_space.append(action_info)
            # ad cache
            elif action_info['outerHTML'] in self.ad_cache:
                if self.ad_cache[action_info['outerHTML']]:
                    action_space.append(action_info)
            # ad
            else:
                elements_index.append(index)
                elements.append(action_info['outerHTML'])
        if len(elements_index):
            outputs = self._recognize_by_discriminator(elements)
            for index in range(len(outputs)):
                res = outputs[index]
                self.ad_cache[elements[index]] = res
                if res:
                    action_space.append(potential_actions[elements_index[index]])
        self.time_cost += time.time() - start_time
        return action_space

    def _train(self):
        self.ad_cache.clear()
        if self.data_cache.train_label is not None:
            train_data = torch.tensor(self.data_cache.train_data).to(self.device)
            train_label = torch.tensor(self.data_cache.train_label).to(self.device)
            self.discriminator.train()

            for epoch in range(self.epochs_per_train):
                outputs = self.discriminator(train_data)
                loss = self.loss_function(outputs, train_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                settings.logger.debug(f"Epoch {epoch + 1}/{self.epochs_per_train}, Loss: {loss.item():.4f}")
                if loss.item() < 0.1:
                    break
        else:
            settings.logger.error("train data is None")

    def check(self, episode_num, continuous_episodes):
        start_time = time.time()
        if self.flag_enable:
            if (episode_num + 1) % 10 == 0:
                self._train()
        else:
            if continuous_episodes >= 6 or episode_num >= 35:  # todo
                self.flag_enable = True
        self.time_cost += time.time() - start_time

    def update(self, data, label):
        if len(label):
            embedding = self.embedding_model.encode(data)
            self.data_cache.update_by_explore(data, label, embedding)
