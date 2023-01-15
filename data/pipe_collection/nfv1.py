from mdlp.data.transform import *
import pandas as pd

IP_MAX = BYTES_MAX = PKTS_MAX = FLOW_DURATION = 4294967295
PORT_MAX = PROTO_7_MAX = 65535
PROTO_MAX = TCP_FLAG_MAX = 255

FEATURE = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
           'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']

FEATURE_MAP = {
    'IPV4_SRC_ADDR': lambda ip: np.float64(sum([int(x) << 8*i for i, x in enumerate(reversed(ip.split('.')))]))/IP_MAX,
    'IPV4_DST_ADDR': lambda ip: np.float64(sum([int(x) << 8*i for i, x in enumerate(reversed(ip.split('.')))]))/IP_MAX,
    'L4_SRC_PORT': lambda port: np.float64(port)/PORT_MAX,
    'L4_DST_PORT': lambda port: np.float64(port)/PORT_MAX,
    'PROTOCOL': lambda proto: np.float64(proto)/PROTO_MAX,
    'L7_PROTO': lambda l7_proto:  np.float64(l7_proto)/PROTO_7_MAX,
    'IN_BYTES': lambda in_b: np.float64(in_b)/BYTES_MAX,
    'OUT_BYTES': lambda out_b: np.float64(out_b)/BYTES_MAX,
    'IN_PKTS': lambda in_p: np.float64(in_p)/PKTS_MAX,
    'OUT_PKTS': lambda out_p: np.float64(out_p)/PKTS_MAX,
    'TCP_FLAGS': lambda tcp_f: np.float64(tcp_f)/TCP_FLAG_MAX,
    'FLOW_DURATION_MILLISECONDS': lambda flow_duration: np.float64(flow_duration)/FLOW_DURATION,
}

DROP_DICT = {
    'IPV4_SRC_ADDR': lambda x: ~x.str.contains("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", regex=True),
    'IPV4_DST_ADDR': lambda x: ~x.str.contains("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", regex=True)
}

binary_base_pipe = [None, FilterColummn(FEATURE + ['Label']), DropValue(DROP_DICT), BalanceData('Label'), TranformValue(FEATURE_MAP)]
nclass_base_pipe = [None, FilterColummn(FEATURE + ['Attack']), DropValue(DROP_DICT), BalanceData('Attack'), TranformValue(FEATURE_MAP)]

def get_pipe(source_type = "CSV", is_binary=True, verbose=True, test_size=None):
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
            data_pipe.add_pipe([FilterColummn(FEATURE), FilterColummn(['Label',])])
        else:
            data_pipe.add_pipe([FilterColummn(FEATURE), FilterColummn(['Attack',])])
            data_pipe.add_pipe([None, OneHotSource('Attack')])
        data_pipe.add_pipe(TrainTestSplit(test_size))

    return data_pipe
