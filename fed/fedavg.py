from mdlp.fed import *
import numpy as np


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
        avg_weight[i] = save_weight[i] + avg_weight[i] * lr_s / total_client_data

    return avg_weight


class FedAvgServer(FedServer):
    def __init__(self, global_model, **kwargs):
        super().__init__(global_model, **kwargs)
    
    def calculate_aggr_value(self, client_param, lr_s, save_weight):
        return avg_aggr(client_param, lr_s, save_weight)

class FedAvgClient(FedClient):
    def __init__(self, local_model, **kwargs):
        super().__init__(local_model, **kwargs)

class FedAvg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def server(self, *args, **kwargs):
        return FedAvgServer(*args, **kwargs)

    def client(self, *args, **kwargs):
        return FedAvgClient(*args, **kwargs)