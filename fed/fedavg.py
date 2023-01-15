from mdlp.fed import *
from mdlp import model
import tensorflow_addons as tfa
import numpy as np
import copy


def avg_aggr(client_param, lr_s, save_weight):
    total_client_data = 0
    avg_weight = []

    for weight in save_weight:
        avg_weight.append(np.zeros_like(weight))

    for weight, sample_count in client_param:
        total_client_data += sample_count

        for i in range(len(avg_weight)):
            avg_weight[i] += sample_count * (weight[i] - save_weight[i])

    for i in range(len(avg_weight)):
        avg_weight[i] = save_weight[i] + \
            avg_weight[i] * lr_s / total_client_data

    return avg_weight


class FedAvgServer(FedServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = None
    
    def initialize(self, config):
        metric = copy.deepcopy(config['compile']['metric'])
        for i in range(len(metric)):
            if metric[i] == 'F1Score':
                metric[i] = tfa.metrics.F1Score(num_classes=config['n_classes'], threshold=0.5)
        
        if getattr(model, config['model']) is not None:
            self.global_model = getattr(model, config['model'])(input_shape=config['feature_shape'], n_classes=config['n_classes'])
        else:
            self.global_model = copy.deepcopy(config['model'])
        
        self.compile_model(loss=config['compile']['loss'], metrics=metric)
        self.update_aggr_value(self.global_model.get_weights())
        self.update_fed_params(self.global_model.get_weights())

    def update(self, config, clients):
        cpa = [c.get_local_params() for c in clients]
        sw = self.get_aggr_value()
        new_aggr_value = self.calculate_aggr_value(cpa, config['lr_s'], sw)
        self.update_fed_params(new_aggr_value)
        self.update_aggr_value(new_aggr_value)
        self.update_model_weights(new_aggr_value)

    def calculate_aggr_value(self, client_param, lr_s, save_weight):
        return avg_aggr(client_param, lr_s, save_weight)
    
    def compile_model(self, *args, **kwargs):
        self.global_model.compile(*args, **kwargs)
    
    def update_model_weights(self, new_weights):
        self.global_model.set_weights(new_weights)

    def evaluate(self, X_test, y_test, **kwargs):
        config = copy.deepcopy(kwargs.pop('config', None))
        return self.global_model.evaluate(X_test, y_test, **config, **kwargs)


class FedAvgClient(FedClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        self.ncd = None

    def initialize(self, config):
        metric = copy.deepcopy(config['compile']['metric'])
        for i in range(len(metric)):
            if metric[i] == 'F1Score':
                metric[i] = tfa.metrics.F1Score(num_classes=config['n_classes'], threshold=0.5)

        if config['compile']['optim'] == 'SGD':
            optim = tf.keras.optimizers.SGD(learning_rate=config['compile']['lr'])
        elif config['compile']['optim'] == 'Adam':
            optim = tf.keras.optimizers.Adam(learning_rate=config['compile']['lr'])
        else:
            optim = None
        if getattr(model, config['model']) is not None:
            self.local_model = getattr(model, config['model'])(input_shape=config['feature_shape'], n_classes=config['n_classes'])
        else:
            self.local_model = copy.deepcopy(config['model'])
        self.compile_model(optim, config['compile']['loss'], metric)

    def update(self, new_values):
        self.update_fed_params(new_values)
        self.update_aggr_value(new_values)
        self.update_model_weights(new_values)
    
    
    def get_ncd(self):
        return self.ncd

    def update_ncd(self, new_ncd):
        self.ncd = new_ncd

    def get_local_params(self):
        return (self.get_model_weights(), self.get_ncd())

    def train(self, dataset, **kwargs):
        config = copy.deepcopy(kwargs.pop('config', None))
        if config==None:
            raise ValueError("`config` keyword argument is required.")
        batch_size = config.pop('batch_size', None)
        X_train = dataset[0]
        y_train = dataset[1]
        if batch_size is None:
            train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0])
        else:
            train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(batch_size)
        self.update_ncd(len(X_train))
        super().train(train_data, **config, **kwargs)

class FedAvg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def server(self, *args, **kwargs):
        return FedAvgServer(*args, **kwargs)

    def client(self, *args, **kwargs):
        return FedAvgClient(*args, **kwargs)
