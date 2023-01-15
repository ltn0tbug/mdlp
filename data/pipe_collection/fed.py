from mdlp.data.transform import *
import pandas as pd

def get_pipe(source_type = "CSV", is_binary=True, test_size=None, n_clients=1, split_attribute=None, verbose=True):
    raw_pipe = []
    if source_type == "CSV":
        raw_pipe.append(FromCSV())
    elif source_type == "DataFrame":
        raw_pipe.append(FromDataFrame())
    elif source_type != "No_Source":
        raise ValueError("`source_type` must be in ['CSV', 'DataFrame', 'No_Source']")

    if test_size is not None:
        raw_pipe.append(SplitDataWithRatio(test_size, split_attribute))
        raw_pipe.append([None, SplitDataWithN(n_clients, split_attribute)])
        raw_pipe.append([[DropColumnWithAttribute(split_attribute), FilterColummn([split_attribute])], [[DropColumnWithAttribute(split_attribute), FilterColummn([split_attribute])]]*n_clients])
        if is_binary is False:
            raw_pipe.append([[None, OneHotSource(split_attribute)], [[None, OneHotSource(split_attribute)]]*n_clients])
    else:
        raw_pipe.append(SplitDataWithN(n_clients, split_attribute))
        raw_pipe.append([[DropColumnWithAttribute(split_attribute), FilterColummn([split_attribute,])]]*n_clients)

    return DataTranformPipeline(raw_pipe, verbose=verbose)