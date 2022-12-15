import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import time
from mdlp.data.utils import list_type, list_shape, list_pipe


class FromCSV:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, source):
        return pd.read_csv(source, **self.kwargs)

# def FromCSV(**kwargs):
#     def call(source):
#         return pd.read_csv(source, **kwargs)
    
#     return call(source)


class FilterColummn:
    def __init__(self, filter_=None, is_exclude=False, **kwargs):
        self.kwargs = kwargs
        self.filter_ = filter_
        self.is_exclude = is_exclude

    def __call__(self, source):
        if self.filter_ is None:
            return source
        if self.is_exclude:
            return source.loc[:, ~source.columns.isin(self.filter_)]
        return source[self.filter_]


class DropNaN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, source):
        return source.dropna(**self.kwargs)


class ReplaceNaN:
    def __init__(self, value=0, **kwargs):
        self.value = value
        self.kwargs = kwargs

    def __call__(self, source):
        if isinstance(self.value, int):
            return source.fillna(**self.kwargs)
        """ Not full implement yet"""
        return source


class TrainTestSplit:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, source):
        return train_test_split(source[0], source[1], **self.kwargs)


class DropValue:
    def __init__(self, condition, **kwargs):
        self.condition = condition
        self.kwargs = kwargs

    def __call__(self, source):
        for col_name, cond in self.condition.items():
            drop_index = source[cond(source[col_name])].index
            source = source.drop(drop_index, **self.kwargs)
        return source


class TranformValue:
    def __init__(self, condition, **kwargs):
        self.condition = condition
        self.kwargs = kwargs

    def __call__(self, source):
        for col_name, cond in self.condition.items():
            source[col_name] = source[col_name].transform(cond, **self.kwargs)
        return source


class ReShape:
    def __init__(self, shape, **kwargs):
        self.kwargs = kwargs
        self.shape = shape

    def __call__(self, source):
        if isinstance(source, pd.DataFrame):
            return np.reshape(source.to_numpy(), self.shape, **self.kwargs)
        if isinstance(source, np.ndarray):
            return np.reshape(source, self.shape, **self.kwargs)
        return tf.reshape(source, self.shape, **self.kwargs)


class ToTensor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, source):
        return tf.convert_to_tensor(source, **self.kwargs)


class ToDataFrame:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, source):
        return pd.DataFrame(source, **self.kwargs)

class ToNumpyArray:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, source):
        if isinstance(source, pd.DataFrame):
            return source.to_numpy(**self.kwargs)
        return pd.array(source, **self.kwargs)


class FlattenSource:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def flatten(source):
        if not isinstance(source, list):
            return [
                source,
            ]
        new_source = []
        for src in source:
            new_source += flatten(src)

        return new_source

    def __call__(self, source):
        if isinstance(source, list):
            return self.flatten(source)
        return source

class OneHotSource:
    def __init__(self, attribute, **kwargs):
        self.kwargs = kwargs
        self.attribute = attribute
    
    def __call__(self, source):
        onehot = OneHotEncoder(sparse=False)
        source_encode = onehot.fit_transform(source[[self.attribute,]])
        return source_encode

class BalanceData:
    def __init__(self, balanced_attribute, shuffle=True):
        self.balanced_attribute = balanced_attribute
        #self.percent = percent
        self.shuffle = shuffle
    
    def __call__(self, source):
        benign = source.groupby(self.balanced_attribute).get_group(0)
        attack = source.groupby(self.balanced_attribute).get_group(1)
        split_size = 0
        if len(benign) < len(attack):
            split_size = len(benign)
            attack = attack.sample(n=split_size, random_state=1)
        else:
            split_size = len(attack)
            benign = benign.sample(n=split_size, random_state=1)
        
        new_source = pd.concat([benign, attack])
        if self.shuffle:
            return new_source.sample(frac=1).reset_index(drop=True)
        return new_source


class ConcatColumn():
    def __call__(self, source):
        return df.concat(source, axis=1)



class GenerateDataSet:
    def __init__(self, shuffle_size=None, batch_size=None, repeat_count=None, **kwargs):
        self.kwargs = kwargs
        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.repeat_count = repeat_count
    

    def __init__(self, source):
        dataset = tf.data.Dataset.from_tensor_slices(source)
        if self.repeat_count is not None:
            dataset = dataset.repeat(repeat_count)
        
        if self.shuffle_size is not None:
            dataset = dataset.shuffle(self.shuffle_size)
        
        if self.batch_size is not None:
            drmd = self.kwargs.get('drop_remainder', False)
            dataset = dataset.batch(self.batch_size, drop_remainder=drmd)


class DataTranformPipeline:
    def __init__(self, pipe=None, verbose=False):
        self.pipe = pipe
        self.verbose = verbose

    def __call__(self, source):

        if self.pipe is None or len(self.pipe) == 0:
            raise ValueError("Pipeline is empty.")

        first_pipe = self.pipe.pop(0)
        if self.verbose:
            x = self.execute_verbose(first_pipe, source)
        else:
            x = self.execute_pipe(first_pipe, source)
        print("-" * 20)

        for p in self.pipe:
            if self.verbose:
                x = self.execute_verbose(p, x)
            else:
                x = self.execute_pipe(p, x)
            print("-" * 20)

        self.pipe.insert(0, first_pipe)

        return x

    def execute_pipe(self, pipe, source):
        if not isinstance(pipe, list):
            if pipe is None:
                return source
            
            if not isinstance(source, list):
                return pipe(source)
            
            if pipe.__class__.__name__ == "TrainTestSplit":
                return pipe(source)

            return [self.execute_pipe(pipe, s) for s in source]

        if not isinstance(source, list):
            return [self.execute_pipe(p, source) for p in pipe]

        if len(pipe) != len(source):
            raise ValueError("Number of pipes and sources are different")

        return [self.execute_pipe(p, s) for p, s in zip(pipe, source)]

    def execute_verbose(self, pipe, source):
        print("Pipe: {}".format(list_pipe(pipe)))

        print("Source type: {}".format(list_type(source)))
        print("Source shape: {}".format(list_shape(source)))

        start_time = time.time()

        x = self.execute_pipe(pipe, source)

        end_time = time.time()
        print("Elapsed_time: {0} ms".format((end_time - start_time) * 1000))

        print("New source type: {}".format(list_type(x)))
        print("New source shape: {}".format(list_shape(x)))

        return x
