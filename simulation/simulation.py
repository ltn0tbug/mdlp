from utils import track
from mdlp.data.pipe_collection import nfv1
from mdlp.data.pipe_collection import fed as fed_pipe
import tensorflow as tf
from mdlp import fed
import yaml
from yaml.loader import SafeLoader
import numpy as np
import copy

@track
def fedsimulator(config_dir, fed_model=None, client_data=None, test_data=None):
    with open(config_dir) as f:
        config = yaml.load(f, Loader=SafeLoader)

    NUM_CLIENTS = config['server']['n_client']
    TEST_SIZE = config['dataset']['test_size']

    if client_data is None or test_data is None:
        nfv1_pipe = nfv1.get_pipe()
        additional_pipe = fed_pipe.get_pipe(source_type = "No_Source", test_size=TEST_SIZE, n_clients=NUM_CLIENTS, split_attribute='Label')
        nfv1_pipe.add_pipe(additional_pipe)
        test_data, client_data = nfv1_pipe(config['dataset']['paths'])
    
    FEATURE_SHAPE = test_data[0].shape[1:]

    mode = config["mode"]
    verbose = config["verbose"]
    config["server"]["feature_shape"] = FEATURE_SHAPE
    config["client"]["feature_shape"] = FEATURE_SHAPE

    print(f'Federated learning mode: {mode}')
    print(f"Server setting: {config['server']}")
    print(f"Client setting: {config['client']}")
    print(f"Dataset setting: {config['dataset']['names']}")
    print(f"X_test data: {test_data[0].shape}")
    print(f"y_test data: {test_data[1].shape}")
    print(f"Client data: {[[cd[0].shape, cd[1].shape] for cd in client_data]}")
    print('-'*20)
    print("Initialize...")
    if isinstance(fed_model, str):
        fed_base = getattr(fed, fed_model)()
    elif fed_model is None:
        fed_base = getattr(fed, config['fed_model'])()
    else:
        fed_base = fed_model
    server = fed_base.server()

    clients = [fed_base.client() for _ in range(NUM_CLIENTS)]
    server.initialize(config['server'])
    for client in clients:
        client.initialize(config['client'])
        fed_params = copy.deepcopy(server.get_fed_params())
        client.update(fed_params)

    print("Done!!!")
    print("Start Simulation")
    for r in range(config['server']['r']):
        print(f"[o] Start Round: {r + 1}")
        for c in range(NUM_CLIENTS):
            print(f"[c{c + 1}] Start Training")
            clients[c].train(client_data[c], config=config['client']['train'])
            print(f"[c{c + 1}] Finish Training")

        print("[s] calculate aggregate value")
        server.update(config['server'], clients)
        print("[s] Update client weights")
        for c in range(len(clients)):
            client.update(server.get_fed_params())
        print("[s] Evaluation")
        server.evaluate(*test_data, config=config['server']['evaluate'])
        print(f"[o] Finish Round {r + 1}")