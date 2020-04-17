import pandas as pd
import numpy as np

import importlib  
script = importlib.import_module("icae.results.02_waveform.01_hdf_rewrite")

def test_summary():
    df1 = pd.DataFrame([[1,2],[3,4]])
    df2 = pd.DataFrame([[5,6],[7,8]])
    correct_summary = df1.append(df2).describe().loc[["count", "mean", "min", "max"]]
    
    summary = script.calc_summary_statistics(df1, None)
    summary = script.calc_summary_statistics(df2, summary)
    
    pd.testing.assert_frame_equal(correct_summary, summary)
    

if __name__ == "__main__":
    test_summary()