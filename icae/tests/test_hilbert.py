import numpy as np
import pandas as pd
import torch
import numpy.testing as npt

from icae.processors.hilbert import Hilbertize, rotate_2d


def test_Hilbertize():
    events, waveforms_per_event, dims, garbage = 10, 3, 4, 4
    rows = events * waveforms_per_event
    index = np.rint(np.linspace(0, events, rows))
    coordinates = np.random.rand(rows,dims)
    coordinate_names = [f"dim_{i}" for i in range(dims)]
    other_columns = np.random.randint(0,100,size=[rows,garbage])
    other_column_names = [f"other_{i}" for i in range(garbage)]
    
    data = np.hstack([coordinates,other_columns])
    df = pd.DataFrame(data,index=index,columns=coordinate_names+other_column_names)
    
    hilbert = Hilbertize(dimension_columns=coordinate_names)
    print(hilbert)
    x = hilbert(df)
    npt.assert_equal(x[other_column_names].values, other_columns)
    print(x)
    

def test_rotate_2d():
    c = np.array([[1.0, 1.0, 10], [0.0, 0.0, 10], [0.0, 1.0, 10]])
    r = rotate_2d(c, np.deg2rad(45))

    npt.assert_almost_equal(r[0], [0, np.sqrt(2), 10])
    npt.assert_almost_equal(r[1], [0, 0, 10])
    npt.assert_almost_equal(r[2], [-np.sqrt(1 / 2), np.sqrt(1 / 2), 10])


if __name__ == "__main__":
    test_rotate_2d()
    test_Hilbertize()

