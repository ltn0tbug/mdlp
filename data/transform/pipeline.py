import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import time
from mdlp.data.utils import list_type, list_shape, list_pipe


class FromCSV:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        return pd.read_csv(source, *self.args, **self.kwargs)


class FromDataFrame:
    def __call__(self, source):
        return source


# def FromCSV(*args, **kwargs):
#     def call(source):
#         return pd.read_csv(source, *args, **kwargs)
#     return call(source)


class FilterColummn:
    def __init__(self, filter_=None, is_exclude=False, *args, **kwargs):
        self.args = args
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
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        return source.dropna(*self.args, **self.kwargs)


class ReplaceNaN:
    def __init__(self, value=0, *args, **kwargs):
        self.value = value
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        if isinstance(self.value, int):
            return source.fillna(*self.args, **self.kwargs)
        """ Not full implement yet"""
        return source


class TrainTestSplit:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        return train_test_split(*source, *self.args, **self.kwargs)


class DropValue:
    def __init__(self, condition, *args, **kwargs):
        self.condition = condition
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        for col_name, cond in self.condition.items():
            drop_index = source[cond(source[col_name])].index
            source = source.drop(drop_index, *self.args, **self.kwargs)
        return source


class TranformValue:
    def __init__(self, condition, *args, **kwargs):
        self.condition = condition
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        for col_name, cond in self.condition.items():
            source[col_name] = source[col_name].transform(
                cond, *self.args, **self.kwargs
            )
        return source


class ReShape:
    def __init__(self, shape, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.shape = shape

    def __call__(self, source):
        if isinstance(source, pd.DataFrame):
            return np.reshape(source.to_numpy(), self.shape, *self.args, **self.kwargs)
        if isinstance(source, np.ndarray):
            return np.reshape(source, self.shape, *self.args, **self.kwargs)
        return tf.reshape(source, self.shape, *self.args, **self.kwargs)


class ToTensor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        return tf.convert_to_tensor(source, *self.args, **self.kwargs)


class ToDataFrame:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        return pd.DataFrame(source, *self.args, **self.kwargs)


class ToNumpyArray:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source):
        if isinstance(source, pd.DataFrame):
            return source.to_numpy(*self.args, **self.kwargs)
        return pd.array(source, *self.args, **self.kwargs)


class FlattenSource:
    def __init__(self, *args, **kwargs):
        self.args = args
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
    def __init__(self, attribute, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.attribute = attribute

    def __call__(self, source):
        onehot = OneHotEncoder(sparse=False)
        source_encode = onehot.fit_transform(
            source[
                [
                    self.attribute,
                ]
            ]
        )
        return source_encode


class BalanceData:
    def __init__(self, balanced_attribute, shuffle=True, random_state=None):
        self.balanced_attribute = balanced_attribute
        self.shuffle = shuffle
        self.random_state = random_state

    def __call__(self, source):
        sgb = source.groupby(self.balanced_attribute)
        ts = [sgb.get_group(x) for x in sgb.groups]
        ll = [len(x) for x in ts]
        midx = ll.index(min(ll))
        split_size = ll[midx]
        new_source = [
            ts[i]
            if i == midx
            else ts[i].sample(n=split_size, random_state=self.random_state)
            for i in range(len(ts))
        ]
        new_source = pd.concat(new_source)
        if self.shuffle:
            return new_source.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
        return new_source


class ConcatColumn:
    def __call__(self, source):
        return df.concat(source, axis=1)


class GenerateDataSet:
    def __init__(
        self, shuffle_size=None, batch_size=None, repeat_count=None, *args, **kwargs
    ):
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
            drmd = self.kwargs.get("drop_remainder", False)
            dataset = dataset.batch(self.batch_size, drop_remainder=drmd)


class SplitDataWithRatio:
    def __init__(
        self, ratio=0.5, shuffle=True, balanced_attribute=None, random_state=None
    ):
        self.ratio = ratio
        self.shuffle = True
        self.balanced_attribute = balanced_attribute
        self.random_state = random_state

    def __call__(self, source):
        if self.balanced_attribute is not None:
            sgb = source.groupby(self.balanced_attribute)
            ts = [sgb.get_group(x) for x in sgb.groups]
            first_source = []
            second_source = []
            for df in ts:
                first_source.append(
                    df.sample(frac=self.ratio, random_state=self.random_state)
                )
                second_source.append(df.drop(first_source[-1].index))
            first_source = pd.concat(first_source)
            second_source = pd.concat(second_source)
            if self.shuffle:
                return [
                    first_source.sample(
                        frac=1, random_state=self.random_state
                    ).reset_index(drop=True),
                    second_source.sample(
                        frac=1, random_state=self.random_state
                    ).reset_index(drop=True),
                ]
            return [
                first_source.reset_index(drop=True),
                second_source.reset_index(drop=True),
            ]
        else:
            if self.shuffle:
                first_source = source.sample(
                    frac=self.ratio, random_state=self.random_state
                )
            else:
                first_source = source.iloc[: int(len(source) * self.ratio)]

            second_source = source.drop(first_source.index)
            return [
                first_source.reset_index(drop=True),
                second_source.reset_index(drop=True),
            ]


class SplitDataWithN:
    def __init__(self, n, shuffle=True, balanced_attribute=None, random_state=None):
        self.n = n
        self.shuffle = shuffle
        self.balanced_attribute = balanced_attribute
        self.random_state = random_state

    def __call__(self, source):
        if self.balanced_attribute is not None:
            source_n = [[] for _ in range(self.n)]
            sgb = source.groupby(self.balanced_attribute)
            ts = [sgb.get_group(x) for x in sgb.groups]
            for df in ts:
                split_size = int(len(df) / self.n)
                for i in range(self.n):
                    source_n[i].append(
                        df.sample(n=split_size, random_state=self.random_state)
                    )
                    df.drop(source_n[i][-1].index)

            for i in range(self.n):
                source_n[i] = pd.concat(source_n[i])

            if self.shuffle:
                return [
                    s.sample(frac=1, random_state=self.random_state).reset_index(
                        drop=True
                    )
                    for s in source_n
                ]
            return [s.reset_index(drop=True) for s in source_n]
        else:
            source_n = []
            split_size = int(len(source) / self.n)
            if self.shuffle:
                for i in range(self.n):
                    source_n.append(
                        source.sample(n=split_size, random_state=self.random_state)
                    )
                    source.drop(source_n[-1].index)
            else:
                for i in range(self.n):
                    source_n.append(source.iloc[i * split_size : (i + 1) * split_size])
            return [s.reset_index(drop=True) for s in source_n]


class DropColumnWithAttribute:
    def __init__(self, attribute):
        self.attribute = attribute

    def __call__(self, source):
        return source.drop(self.attribute, axis=1)


class DataTranformPipeline:
    def __init__(self, pipe=None, verbose=False):
        self.pipe = pipe
        self.verbose = verbose

    def __call__(self, source):

        if self.pipe is None or len(self.pipe) == 0:
            raise ValueError("Pipeline is empty.")

        first_pipe = self.pipe.pop(0)
        first_pipe = (
            first_pipe[0]
            if isinstance(first_pipe, list) and len(first_pipe) == 1
            else first_pipe
        )
        source = source[0] if isinstance(source, list) and len(source) == 1 else source
        if self.verbose:
            x = self.execute_verbose(first_pipe, source)
        else:
            x = self.execute_pipe(first_pipe, source)

        x = x[0] if isinstance(x, list) and len(x) == 1 else x
        print("-" * 20)

        for p in self.pipe:
            if self.verbose:
                x = self.execute_verbose(p, x)
            else:
                x = self.execute_pipe(p, x)
            x = x[0] if isinstance(x, list) and len(x) == 1 else x
            print("-" * 20)

        self.pipe.insert(0, first_pipe)

        return x

    def __str__(self):
        return str([list_pipe(p) for p in self.pipe])

    def get_raw_pipe(self):
        return self.pipe

    def add_pipe(self, pipe):
        if isinstance(pipe, DataTranformPipeline):
            self.pipe.extend(pipe.get_raw_pipe())
        else:
            self.pipe.append(pipe)

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
        elapsed = end_time - start_time
        print(
            "Elapsed_time: {0}".format(
                time.strftime(
                    "%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15],
                    time.gmtime(elapsed),
                )
            )
        )

        print("New source type: {}".format(list_type(x)))
        print("New source shape: {}".format(list_shape(x)))

        return x
