import tensorflow as tf


class FedBase:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def server(self, *args, **kwargs):
        pass

    def client(self, *args, **kwargs):
        pass

    def simulation(self):
        pass


class FedServer:
    def __init__(self, global_model, **kwargs):
        self.global_model = global_model
        self.aggr_value = global_model.get_weights()
        init_client_identity = kwargs.get("init_client_identity", [])
        init_client_config = kwargs.get("init_client_config", [])
        self.client = (
            init_client_identity if isinstance(init_client_identity, list) else []
        )

    def calculate_aggr_value(self):
        pass

    def get_aggr_value(self):
        return self.aggr_value

    def update_aggr_value(self, new_value):
        self.aggr_value = new_value

    def reinitialize_model_weight(self):
        import keras.backend as K

        session = K.get_session()
        for layer in self.global_model.layers:
            if hasattr(layer, "kernel_initializer"):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, "bias_initializer"):
                layer.bias.initializer.run(session=session)

    def reinitialize_model_weight_and_aggr_value(self):
        self.reinitialize_model_weight()
        self.update_aggr_value(self.global_model.get_weights())

    def remove_client(self, client_indentity):
        pass

    def add_client(self, client_indentity):
        self.client.append(client_indentity)

    def up_client_identity(self, old_identity, new_identity):
        pass

    def send_params(self, client_identity):
        """Not fully implemented yet"""

        if isinstance(client_identity, Identity):
            client_identity.send(self.aggr_func)


class FedClient:
    def __init__(self, local_model, server_identity, **kwargs):
        self.local_model = local_model
        self.server_identity = server_identity
        self.kwargs = kwargs

    def compile_model(self, model_config):
        self.local_model.compile(**model_config)
    
    def update_model_weight(self, new_weight):
        self.local_model.set_weights(new_weight)
    
    def save_model(self, save_path):
        self.local_model.save(save_path)
    
    def save_model_weight(self, save_path):
        self.local_model.save_weights(save_path)

    def get_fed_params(self):
        pass

    def get_global_config(self):
        pass

    def post_local_params(self):
        pass

    def train(self, dataset, train_config):
        self.local_model.fit(dataset, **train_config)

    def evaluate(self, dataset):
        self.local_model.evaluate(dataset)

    def reinitialize_model(self):
        import keras.backend as K

        session = K.get_session()
        for layer in self.global_model.layers:
            if hasattr(layer, "kernel_initializer"):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, "bias_initializer"):
                layer.bias.initializer.run(session=session)

    def update_server_identity(self, new_server_identity=None):
        pass


class Identity:
    def __init__(self, id_, **kwargs):
        self.id_ = id_
        self.kwargs = kwargs

    def send(self, param):
        pass

    def get_id(self):
        return self.id_

