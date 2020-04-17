# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---
"""
Extracts some event attributes (waveforms, om, string, etc.) from
IceCube MC files and saves them as pandas dataframes (HDF).
Use this file non-interactively with CL bindings (fire).
"""

# +
# !/cvmfs/icecube.opensciencegrid.org/py2-v3/RHEL_7_x86_64/bin/python
from __future__ import (
    division,
)  # , with_statement,generators, unicode_literals,print_function, absolute_import
import numpy as np
import xarray as xr
import os, sys
from I3Tray import *
from icecube import icetray, dataclasses, dataio
from icecube.icetray import I3Units


from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

from icae.tools import performance
from icae.tools.config_loader import config, mkdir_p
import icae.tools.icetray_mmeier.Waveform_Calibration as wavecal

import pandas as pd



class ExtractWaveforms(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter("output_file", "where the npz files are saved", None)
        self.info = []
        self.waveforms = []
        self.frame_number = -1
        self.output_file = ""

    def Configure(self):
        self.output_file = self.GetParameter("output_file")

    def Physics(self, frame):
        self.frame_number += 1
        if frame.Has("CalibratedWaveforms"):
            doms = frame["CalibratedWaveforms"]
            geometry = frame["I3Geometry"].omgeo
            for dom in doms:
                count_important_events_in_this_dom = -1
                for event in dom.data():
                    # there are three types of waveform data: len=(3,128,256)
                    # a short investigation showed that only for 128 there are
                    # actual peaks. So we only take those.
                    if len(event.waveform) == 128:
                        count_important_events_in_this_dom += 1

                        om = dom.key().om
                        string = dom.key().string
                        pos = geometry[dom.key()].position
                        x, y, z = pos.x, pos.y, pos.z

                        time = event.time

                        self.waveforms.append(
                            [
                                self.frame_number,
                                string,
                                om,
                                time,
                                x,
                                y,
                                z,
                                event.waveform,
                            ]
                        )

        self.PushFrame(frame)

    def Finish(self):
        print("writing data frame to disk")
        self.waveforms = np.asarray(self.waveforms)
        last_index_column = 4
        index = pd.MultiIndex.from_arrays(
            self.waveforms[:, 0:last_index_column].T,
            names=["frame", "string", "om", "starting_time"],
        )
        columns = ["x", "y", "z"]
        columns.extend(["t=%d" % i for i in range(128)])

        # convert df cells with np arrays to a big numpy array:
        wfs = np.asarray(self.waveforms[:, -1].tolist(), dtype=np.float32)
        other_columns = np.asarray(
            self.waveforms[:, last_index_column:-1], dtype=np.float32
        )
        data = np.hstack((other_columns, wfs))

        df = pd.DataFrame(data, index=index, columns=columns)
        df.to_hdf(
            self.output_file,
            config.data.hdf_key,
            data_columns=True,
            format="table",
            complevel=1,
            complib="bzip2",
        )


def main(input_files, output_file, count_frames=10):
    print("Creating I3Tray for", input_files)
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", Filenamelist=input_files)
    tray.AddSegment(wavecal.CalibrationAndCleaning, "CalibAndCleaning")
    tray.AddModule(ExtractWaveforms, "my_module", output_file=output_file)
    tray.Add("Dump")
    tray.Execute(count_frames)


def extract(output_file, input_file, count_frames=10000):
    print("file to extract:", input_file)
    files = [config.data.gcd_file, input_file]
    print("extracting", files)
    main(files, output_file, count_frames=count_frames)


if __name__ == "__main__":
    in_files = glob(config.data.tau_files)
    out_path = config.data.scratch
    mkdir_p(out_path)

    count_files = config.data.count_files_to_process
    batches = np.array_split(
        in_files[:count_files],
        count_files // config.machine.cpu_cores_for_MC_extraction,
    )
    batches = [list(i) for i in batches]  # tools.preformance doesn't like numpy arrays

    def do(file):
        extract(
            out_path + file.replace("/", "-") + ".hdf", file, count_frames=100000000
        )

    print("Extracting", len(batches), "batches a", len(batches[0]), "files")
    from time import sleep

    sleep(3)
    for i, b in enumerate(batches):
        print("Processing batch %d of %d" % (i, len(batches)))
        performance.multiprocess(do, b)
