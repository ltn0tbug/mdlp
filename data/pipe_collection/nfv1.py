from mdlp.data.transform import *
import pandas as pd
from scipy.special import erf

IP_MAX = BYTES_MAX = PKTS_MAX = FLOW_DURATION = 4294967295
PORT_MAX = PROTO_7_MAX = 65535
PROTO_MAX = TCP_FLAG_MAX = 255

FEATURE = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]

FEATURE_MAP = {
    "IPV4_SRC_ADDR": lambda ip: np.float64(
        sum([int(x) << 8 * i for i, x in enumerate(reversed(ip.split(".")))])
    )
    / IP_MAX,
    "IPV4_DST_ADDR": lambda ip: np.float64(
        sum([int(x) << 8 * i for i, x in enumerate(reversed(ip.split(".")))])
    )
    / IP_MAX,
    "L4_SRC_PORT": lambda port: np.float64(port) / PORT_MAX,
    "L4_DST_PORT": lambda port: np.float64(port) / PORT_MAX,
    "PROTOCOL": lambda proto: np.float64(proto) / PROTO_MAX,
    "L7_PROTO": lambda l7_proto: np.float64(l7_proto) / PROTO_7_MAX,
    "IN_BYTES": lambda in_b: np.float64(in_b) / BYTES_MAX,
    "OUT_BYTES": lambda out_b: np.float64(out_b) / BYTES_MAX,
    "IN_PKTS": lambda in_p: np.float64(in_p) / PKTS_MAX,
    "OUT_PKTS": lambda out_p: np.float64(out_p) / PKTS_MAX,
    "TCP_FLAGS": lambda tcp_f: np.float64(tcp_f) / TCP_FLAG_MAX,
    "FLOW_DURATION_MILLISECONDS": lambda flow_duration: np.float64(flow_duration)
    / FLOW_DURATION,
}

FEATURE_V2 = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]

KW_MAP = {"FLOW_DURATION_MILLISECONDS": 600, "IN_PKTS": 20, "OUT_PKTS": 20, "IN_BYTES": 200, "OUT_BYTES": 200}
FEATURE_MAP_V2 = {
    "IPV4_SRC_ADDR": lambda ip: np.float64(
        sum([int(x) << 8 * i for i, x in enumerate(reversed(ip.split(".")))])
    )
    / IP_MAX,
    "IPV4_DST_ADDR": lambda ip: np.float64(
        sum([int(x) << 8 * i for i, x in enumerate(reversed(ip.split(".")))])
    )
    / IP_MAX,
    "L4_SRC_PORT": lambda port: np.float64(port) / PORT_MAX,
    "L4_DST_PORT": lambda port: np.float64(port) / PORT_MAX,
    "PROTOCOL": lambda proto: np.float64(proto) / PROTO_MAX,
    "L7_PROTO": lambda l7_proto: np.float64(l7_proto) / PROTO_7_MAX,
    "IN_BYTES": lambda in_b: np.float64(erf(in_b/KW_MAP['IN_BYTES'])),
    "OUT_BYTES": lambda out_b: np.float64(erf(out_b/KW_MAP['OUT_BYTES'])),
    "IN_PKTS": lambda in_p: np.float64(erf(in_p/KW_MAP['IN_PKTS'])),
    "OUT_PKTS": lambda out_p: np.float64(erf(out_p/KW_MAP['OUT_PKTS'])),
    "TCP_FLAGS": lambda tcp_f: np.float64(tcp_f) / TCP_FLAG_MAX,
    "FLOW_DURATION_MILLISECONDS": lambda flow_duration: np.float64(erf(flow_duration/KW_MAP['FLOW_DURATION_MILLISECONDS'])),
}

DROP_DICT = {
    "IPV4_SRC_ADDR": lambda x: ~x.str.contains(
        "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", regex=True
    ),
    "IPV4_DST_ADDR": lambda x: ~x.str.contains(
        "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", regex=True
    ),
}

binary_base_pipe = [
    None,
    FilterColummn(FEATURE + ["Label"]),
    DropValue(DROP_DICT),
    BalanceData("Label"),
    TranformValue(FEATURE_MAP),
]
nclass_base_pipe = [
    None,
    FilterColummn(FEATURE + ["Attack"]),
    DropValue(DROP_DICT),
    BalanceData("Attack"),
    TranformValue(FEATURE_MAP),
]

binary_base_pipe_v2 = [
    None,
    FilterColummn(FEATURE_V2 + ["Label"]),
    BalanceData("Label"),
    TranformValue(FEATURE_MAP_V2),
]
nclass_base_pipe_v2 = [
    None,
    FilterColummn(FEATURE_V2 + ["Attack"]),
    BalanceData("Attack"),
    TranformValue(FEATURE_MAP_V2),
]


def get_pipe(source_type="CSV", is_binary = True, verbose = True, test_size = None):
    if is_binary:
        data_pipe = DataTranformPipeline(binary_base_pipe.copy(), verbose=verbose)

    else:
        data_pipe = DataTranformPipeline(nclass_base_pipe.copy(), verbose=verbose)

    
    if source_type == "CSV":
        data_pipe.pipe[0] = FromCSV()
    elif source_type == "DataFrame":
        data_pipe.pipe[0] = FromDataFrame()
    elif source_type != "No_Source":
        raise ValueError("`source_type` must be in ['CSV', 'DataFrame', 'No_Source']")

    if test_size is not None:
        if is_binary:
            data_pipe.add_pipe(
                [
                    FilterColummn(FEATURE),
                    FilterColummn(
                        [
                            "Label",
                        ]
                    ),
                ]
            )
        else:
            data_pipe.add_pipe(
                [
                    FilterColummn(FEATURE),
                    FilterColummn(
                        [
                            "Attack",
                        ]
                    ),
                ]
            )
            data_pipe.add_pipe([None, OneHotSource("Attack")])
        data_pipe.add_pipe(TrainTestSplit(test_size))

    return data_pipe

def get_pipe_v1(source_type="CSV", is_binary=True, verbose=True, test_size=None):
    return get_pipe(source_type, is_binary, verbose, test_size)


"""
@article{raskovalov2022investigation,
  title={Investigation and rectification of NIDS datasets and standratized feature set derivation for network attack detection with graph neural networks},
  author={Raskovalov, Anton and Gabdullin, Nikita and Dolmatov, Vasily},
  journal={arXiv preprint arXiv:2212.13994},
  year={2022}
}
"""

def get_pipe_v2(source_type="CSV", is_binary=True, verbose=True, test_size=None, include_ip = False):
    if is_binary:
        data_pipe = DataTranformPipeline(binary_base_pipe_v2.copy(), verbose=verbose)
        if include_ip:
            data_pipe.pipe[1] = FilterColummn(FEATURE + ["Label"])
    else:
        data_pipe = DataTranformPipeline(nclass_base_pipe_v2.copy(), verbose=verbose)
        if include_ip:
            data_pipe.pipe[1] = FilterColummn(FEATURE + ["Attack"])

    if source_type == "CSV":
        data_pipe.pipe[0] = FromCSV()
    elif source_type == "DataFrame":
        data_pipe.pipe[0] = FromDataFrame()
    elif source_type != "No_Source":
        raise ValueError("`source_type` must be in ['CSV', 'DataFrame', 'No_Source']")

    if test_size is not None:
        if is_binary:
            data_pipe.add_pipe(
                [
                    FilterColummn(FEATURE_V2),
                    FilterColummn(
                        [
                            "Label",
                        ]
                    ),
                ]
            )
        else:
            data_pipe.add_pipe(
                [
                    FilterColummn(FEATURE_V2),
                    FilterColummn(
                        [
                            "Attack",
                        ]
                    ),
                ]
            )
            data_pipe.add_pipe([None, OneHotSource("Attack")])
        data_pipe.add_pipe(TrainTestSplit(test_size))

    return data_pipe