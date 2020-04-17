import os
import time

import scipy

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal
from typing import Tuple, Dict
import yaml
from joblib import Memory
from tqdm import tqdm

import sys

sys.path.append("..")

import icae.tools.performance as toolperf
import icae.toy.waveformMC.numerical as numerical


def ident(*args):
    return args


class Generator:
    """Produce MC waveforms with the generate_* function. Those are saved to df_waveforms, which can be saved by calling save()."""

    # memory = Memory(cache_dir, verbose = 0, compress = 7)
    wf_columns = list("t=%d" % i for i in range(128))
    info_columns = [
        "MC_type",
        "MC_name",
        "d",
        "η",
        "photon_count",
        "t_min",
        "t_max",
        "glitch_params_ID",
    ]
    columns = info_columns + wf_columns

    def __init__(self, ds: Tuple, ηs: Tuple, event_size: stats.rv_continuous):
        self.ds = ds
        self.ηs = ηs
        self.event_size = event_size
        self.event_size_distribution_name = str(event_size.__class__)
        self.event_size_mean = float(event_size.mean())
        self.event_size_std = float(event_size.std())

        self.df_waveforms = pd.DataFrame(columns=self.columns)
        self.df_parameters = pd.DataFrame(columns=["MC_type", "MC_name","count", "params"])

        self.t_max = 420e-9  # sec. See https://arxiv.org/abs/0810.4930v2
        self.tqdm=True

    def save(self, filename, table_name=None):
        self.df_waveforms.to_hdf(
            filename, table_name+"waveforms", complevel=3, complib="blosc:zstd",format='table'
        )
        self.df_parameters.to_hdf(
            filename, table_name+"parameters", complevel=3, complib="blosc:zstd"
        )

    def waveforms_from_pdf(self, times, pdf, sizes):
        waveforms = []
        t_range = times.min(), times.max()
        for size in sizes:
            sample_times = np.random.choice(times, p=pdf / pdf.sum(), size=size)
            hist, _ = np.histogram(sample_times, bins=128, range=t_range)
            waveforms.append(hist)
        return waveforms

    def waveform_from_pdf(self, times, pdf, photons):
        t_range = times.min(), times.max()
        sample_times = np.random.choice(times, p=pdf / pdf.sum(), size=photons)
        waveform, _ = np.histogram(sample_times, bins=128, range=t_range)
        return waveform

    def average_waveform_from_pdf(self, pdf):
        samples = 100000
        waveform = (scipy.signal.resample(pdf, 128) * samples).astype('int')
        return waveform

    def generate_from_func(self, size, pdf_func, glitch_ID, pdf_transform=ident):
        ds = np.random.choice(self.ds, size)
        ηs = np.random.choice(self.ηs, size)
        photons = np.round(self.event_size.rvs(size=size))
        output = np.empty((len(photons), len(self.columns)), dtype="object")
        name = self.df_parameters.loc[glitch_ID, "MC_name"]
        type = self.df_parameters.loc[glitch_ID, "MC_type"]

        # generate all pdfs once to speed things up
        times_and_pdfs = {}
        for params in tqdm(set(zip(ds, ηs)), "Generating PDFs", disable=not self.tqdm):
            times_and_pdfs[params] = pdf_func(d=params[0], η=params[1])

        for i, (d, η, count) in tqdm(
            enumerate(zip(ds, ηs, photons)), "Drawing random samples", total=len(photons), disable=not self.tqdm
        ):
            times, pdf = pdf_transform(*times_and_pdfs[(d, η)])
            count = max(int(count), 1)
            waveform = self.waveform_from_pdf(times, pdf, photons=count)

            # ["MC_type", "MC_name", "d", "η", "photon_count","t_min","t_max", "glitch_params_ID"]
            output[i, :8] = [type, name, d, η, count, 0, self.t_max, glitch_ID]
            output[i, 8:] = waveform

        # for some reason pandas needs to infer twice to infer everything correctly
        output = pd.DataFrame(columns=self.columns, data=output)
        self.df_waveforms = self.df_waveforms.append(output, ignore_index=True).infer_objects().astype({'MC_name':'category'})
        return output

    def _record_glitch_params(self, id, name,size, params):
        self.df_parameters = self.df_parameters.append(
            {"MC_type": id, "MC_name": name, "count":size,"params": params}, ignore_index=True
        )
        return len(self.df_parameters) - 1

    def pdf_valid(self, d, η):
        times,pdf = numerical.pandel_convolved_PDF(self.t_max, d, η)
        return times, pdf/pdf.sum()

    def generate_valid(self, size):
        glitch_ID = self._record_glitch_params(0, "valid",size, {})
        return self.generate_from_func(size, self.pdf_valid, glitch_ID)

    def pdf_gaussian_reshaped(self, d, η, mean, std, impact):
        """mean, std in relative units (mean=1 means a gaussian centered at the right-hand side of the plot).
        impact:  relative difference between peak intensities (1: second peak has the same size. 0.5 second peak at half size)"""
        times, pdf = numerical.pandel_convolved_PDF(self.t_max, d, η)
        norm = scipy.stats.norm.pdf(np.linspace(0, 1, num=len(times)), mean, std)
        modified_pdf = pdf / pdf.sum() + norm / norm.sum() * impact
        return times, modified_pdf / modified_pdf.sum()

    def generate_gaussian_reshaped(self, size, mean, std, impact):
        """Generate DOM waveforms from Pandel + Gauss.
        
        Args:
            size ([type]): number of events to generate
            mean ([type]): [0,1] position of Gaussian within waveform
            std ([type]): std of Gaussian
            impact ([type]): hight of Gaussian
        """
        glitch_ID = self._record_glitch_params(
            1, "gaussian_reshaped", size,{"mean": mean, "std": std, "impact": impact}
        )

        pdf_func = lambda d,η: self.pdf_gaussian_reshaped(d,η,mean,std,impact)

        return self.generate_from_func(size, pdf_func, glitch_ID)


    def pdf_double_pulse(self, d, η, shift, photon_count_difference):
        """shift: time separation between peaks
        photon_count_difference: relative difference between peak intensities (1: second peak has the same size. 0.5 second peak at half size)"""
        times, pdf = numerical.pandel_convolved_PDF(self.t_max, d, η)
        shift = np.searchsorted(times-times.min(), shift)
        shifted = np.roll(pdf, shift)
        shifted[:shift] = 0
        modified_pdf = pdf + shifted * photon_count_difference
        return times, modified_pdf / modified_pdf.sum()
    
    
    
    def generate_double_pulse(self, size, seperation, photon_count_difference):
        glitch_ID = self._record_glitch_params(
            2,
            "double pulse",
            size,
            {
                "separation": seperation,
                "photon_count_difference": photon_count_difference,
            },
        )

        pdf_func = lambda d, η: self.pdf_double_pulse(d,η,seperation,photon_count_difference)

        return self.generate_from_func(size, pdf_func, glitch_ID)
