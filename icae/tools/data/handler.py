"""manages access and modification of larger-than-RAM datasets"""
import dask.dataframe as dd
import dask
import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self, file, readonly=True, default_index="frame", hdf_key="table"):
        self.index = "frame"
        self.hdf_key = "table"

        self.mode = "r" if readonly else "w"
        self.store = pd.HDFStore(file, mode)
        self.dd = dd.read_hdf(file, self.hdf_key, mode="r")

    def __getitem__(self, index):
        return self.get_range(index, index + 1)

    def get_range(self, start, stop, index_name=None):
        assert start > stop
        if not index_name:
            index_name = self.index
        if start + 1 == stop:
            selector = f"{index_name} == start"
        else:
            selector = f"{index_name} >= start & {index_name} < stop"
        return self.store.select(self.hdf_key, selector)
    
    def range(self, column):
        """returns min and max of column"""
        borders = [self.dd.min(column), self.dd.max(column)]
        return dask.compute(borders)

    def __enter__(self):
        pass

    def __exit__(self):
        self.store.close()
