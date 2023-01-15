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
    def __init__(self, *args, **kwargs):
        self.fed_params = None
        self.aggr_value = None
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

    def get_fed_params(self):
        return self.fed_params

    def update_fed_params(self, new_params):
        self.fed_params = new_params

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
    def __init__(self, local_model=None, server_identity=None, **kwargs):
        self.local_model = local_model
        self.server_identity = server_identity
        self.kwargs = kwargs
        self.fed_params = None
        self.aggr_value = None

    def compile_model(self, *args, **kwargs):
        self.local_model.compile(*args, *kwargs)

    def update_model_weights(self, new_weights):
        self.local_model.set_weights(new_weights)

    def save_model(self, save_path):
        self.local_model.save(save_path)

    def save_model_weight(self, save_path):
        self.local_model.save_weights(save_path)

    def get_model_weights(self):
        return self.local_model.get_weights()

    def get_aggr_value(self):
        return self.aggr_value

    def update_aggr_value(self, new_value):
        self.aggr_value = new_value

    def get_fed_params(self):
        return self.fed_params

    def update_fed_params(self, new_params):
        self.fed_params = new_params

    def retrieve_fed_params(self):
        pass

    def retrieve_global_config(self):
        pass

    def get_local_params(self):
        pass

    def post_local_params(self):
        pass

    def train(self, dataset, *args, **kwargs):
        self.local_model.fit(dataset, *args, **kwargs)

    def evaluate(self, X_test, y_test, *args, **kwargs):
        self.local_model.evaluate(X_test, y_test, *args, **kwargs)

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
