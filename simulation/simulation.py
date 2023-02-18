from mdlp.utils import track
import tensorflow as tf
from mdlp import fed
import yaml
from yaml.loader import SafeLoader
import numpy as np
import copy


@track
def fedsimulator(config_dir, test_data, client_data, fed_model=None, keras_model=None):
    simulation_return = {}

    with open(config_dir) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    if "evaluation_history" in config['simulation_return']:
        simulation_return['evaluation_history'] = {}

    NUM_CLIENTS = config["server"]["n_clients"]
    FEATURE_SHAPE = config["dataset"]["feature_shape"]
    N_CLASSES = config["dataset"]["n_classes"]

    if FEATURE_SHAPE != test_data[0].shape[1:]:
        print(f"[W] Input shape {(test_data[0].shape[1:])} is different from config dataset shape {tuple(FEATURE_SHAPE)}. Choose input shape {(test_data[0].shape[1:])} for training.")
        FEATURE_SHAPE = test_data[0].shape[1:]

    mode = config["mode"]
    verbose = config["verbose"]
    config["server"]["feature_shape"] = FEATURE_SHAPE
    config["client"]["feature_shape"] = FEATURE_SHAPE
    config["server"]["n_classes"] = N_CLASSES
    config["client"]["n_classes"] = N_CLASSES

    if keras_model is not None:
        config["server"]["model"] = keras_model
        config["client"]["model"] = keras_model

    print(f"Federated learning mode: {mode}")
    print(f"Server setting: {config['server']}")
    print(f"Client setting: {config['client']}")
    print(f"Dataset setting: {config['dataset']['names']}")
    print(f"X_test data: {test_data[0].shape}")
    print(f"y_test data: {test_data[1].shape}")
    print(f"Client data: {[[cd[0].shape, cd[1].shape] for cd in client_data]}")
    print("-" * 20)
    print("Initialize...")
    if isinstance(fed_model, str):
        fed_base = getattr(fed, fed_model)()
    elif fed_model is None:
        fed_base = getattr(fed, config["fed_model"])()
    else:
        fed_base = fed_model
    server = fed_base.server()

    clients = [fed_base.client() for _ in range(NUM_CLIENTS)]
    server.initialize(config["server"])
    for client in clients:
        client.initialize(config["client"])
        fed_params = copy.deepcopy(server.get_fed_params())
        client.update(fed_params)

    print("Done!!!")
    print("Start Simulation")
    for r in range(config["server"]["r"]):
        print(f"[o] Start Round: {r + 1}")
        for c in range(NUM_CLIENTS):
            print(f"[c{c + 1}] Start Training")
            clients[c].train(client_data[c], config=config["client"]["train"])
            print(f"[c{c + 1}] Finish Training")

        print("[s] calculate aggregate value")
        server.update(config["server"], clients)
        print("[s] Update client weights")
        for c in range(NUM_CLIENTS):
            client[c].update(server.get_fed_params())
        print("[s] Evaluation")
        evaluation_history = server.evaluate(*test_data, config=config["server"]["evaluate"])
        if "evaluation_history" in config['simulation_return']:
            simulation_return['evaluation_history'][f'round{r+1}'] = copy.deepcopy(evaluation_history)
        print(f"[o] Finish Round {r + 1}")
    
    if 'fed_server' in config['simulation_return']:
        simulation_return['fed_server'] = server
    return simulation_return
