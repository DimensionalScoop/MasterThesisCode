"""Functions to generate the Pandel PDF itself"""

import numpy as np
import numpy.testing as test
import scipy as sy
import scipy.stats as stats
import scipy.special as special
import scipy.signal as signal
from scipy import constants as con
from typing import Tuple, List
from joblib import Memory
import collections
from warnings import warn

_memory = Memory("cache/", verbose=0, compress=7)


def pandel_PDF(t: np.ndarray, d: float, η: float) -> np.ndarray:
    """
    Calculates the Pandel PDF.
    Refer to ` Stacked searches for HE nu from blazars with IceCube, Kai Schatto, Disertation` for the algorithm.
    Note that the returned PDF doesn't necessarily integrate to 1 as
    a) the distribution peaks at t=0 but can't be evaluated at t=0 and
    b) the distribution has a very long tail.
    
    :param t: points in time where the PDF should be evaluated at.
    Measured as residual times: The difference between measured time t_hit and
    geometrical, unscattered time.
    :param d: actual distance DOM-photon origin
    :param η: angle between PMT axis and photon track (see Fig 3.1 Schatto)
    :return: Values of the PDF at the specified times `t`.
    """
    assert min(t) > 0, "the gamma function can't handle t=0!"

    # taken from Katz-Spiering p. 22, for polar ice 2.2-2.5 depth
    λ = 40  # m; scattering length
    λ_a = 150  # m; absorption length

    # from Wikipedia
    c_medium = con.c / 1.31  # n ; speed of light in ice
    # taken from Stacked searches for HE nu from blazars with IceCube, Kai Schatto, Disertation
    τ = 557e-9  # s
    a_1 = 0.84
    a_0 = 3.1 - 3.9 * np.cos(η) + 4.6 * np.cos(η) ** 2  # m
    d_eff = a_0 + a_1 * d  # m; effective distance of photon in ice

    def N(d, τ):
        """(3.10) Schatto"""
        r = np.exp(-d / λ_a) * (1 + τ * c_medium / λ_a) ** (-d / λ)
        return r

    def p(t_res, d, τ):
        """(3.10) Schatto"""
        factor = (
            1 / N(d, τ) * τ ** (-d / λ) * t_res ** (d / λ - 1) / special.gamma(d / λ)
        )
        exponent = -t_res * (1 / τ + c_medium / λ_a) - d / λ_a
        return factor * np.exp(exponent)

    return p(t, d_eff, τ)


def _gauss(x, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x ** 2) / sigma ** 2 / 2)


def _assert_is_normed(distribution, x=None, dx=None):
    if __debug__:
        test.assert_allclose(np.trapz(distribution, x=x, dx=dx), 1, 1e-3)


@_memory.cache
def pandel_convolved_PDF(
    t_max: float, d: float, η: float, steps=10000, time_jitter_std=2.7e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the Pandel PDF and convolves it with a Gaussian to account for PMT clock jitter.
    The maximum of the PDF is kept at t_res = 0.
    Time jitter default is according to [Calibration and Characterization of the IceCube Photomultiplier Tube]"""

    mask_below = 1e-9
    acceptable_PDF_error = 1e-7
    start_t = 1e-12 # << 420 ns

    step_size = t_max / steps
    ts = np.linspace(start_t, t_max, steps)

    pdf = pandel_PDF(ts, d, η)
    
    gauss_range = time_jitter_std*4 # 4*std means a coverage of 1 - 1/15787
    if __debug__ and 2*gauss_range/step_size>10000:
        warn(f"pandel_convolved_PDF: `time_jitter_std >> t_max` which leads to a lot of overhead when doing the convolution. ({2*gauss_range/step_size})")
    gx = np.arange(-gauss_range, gauss_range, step_size)
    kernel = _gauss(gx, time_jitter_std)

    # if this assertion fails, the `gx` previously calculated was too small
    _assert_is_normed(kernel, dx=gx[1]-gx[0])

    conv = np.convolve(pdf, kernel, mode="full")
    # `pandel_PDF` won't produce a normed PDF in all cases
    conv = conv / np.trapz(conv, dx=step_size)

    # remove everything that isn't probable to occur anyways
    min_val = max(conv) * mask_below
    mask = conv > min_val
    assert (
        np.sum(conv[~mask])/np.sum(conv) < acceptable_PDF_error
    ), f"Too many small values in folded PDF! ({np.sum(~mask)} times masked, which are {np.sum(conv[~mask])/np.sum(conv)*100} % of the PDF)"
    conv = conv[mask]  # distribution is unimodal

    # the PDF was only calculated between 0 and t_max
    if len(conv)*step_size > t_max:
        conv = conv[:int(t_max/step_size)]

    # generate new time axis that is centered on the maximum
    max_pos = np.argmax(conv)
    steps = conv.size
    numeric_fudge = 0.01  # needed to generate exactly conv.size entries.
    t = np.arange(
        -max_pos * step_size,
        (conv.size - max_pos - numeric_fudge) * step_size,
        step_size,
    )

    assert t.shape == conv.shape
    return t, conv


class Pandel(stats.rv_continuous):
    """XXX: There is no unfrozen version of this class!"""

    def __init__(self, d, η, t_res_max, **kwargs):
        self.name = "Pandel distribution"
        self.d = d
        self.η = η
        self.t_res_max = t_res_max
        self.xs, self.cached_PDF = pandel_convolved_PDF(self.t_res_max, d, η)
        width = self.xs[1] - self.xs[0]
        self.cached_CDF = np.cumsum(self.cached_PDF * width)

        # always do this manually
        if "a" in kwargs:
            kwargs.pop("a")
        if "b" in kwargs:
            kwargs.pop("b")
        super().__init__(a=min(self.xs), b=max(self.xs), **kwargs)

    def __getstate__(self):
        # include our custom 'frozen' parameters
        super_state = super().__getstate__()
        super_state[0]["d"] = self.d
        super_state[0]["η"] = self.η
        super_state[0]["t_res_max"] = self.t_res_max
        return super_state

    def __hash__(self):
        return hash(self.d) ^ hash(self.η) ^ hash(self.t_res_max)

    def _cdf(self, x, *args):
        if not isinstance(x, collections.Iterable):
            return self._do_cdf(x)
        else:
            return [self._do_cdf(i) for i in x]

    def _do_cdf(self, x):
        # TODO(Max): improve performance with multiprocessing
        # means that x is too small/big to fall into the calculated range
        if x < self.a:
            return 0
        if x > self.b:
            return 1

        indices = Pandel._find_index(x, self.xs)
        return Pandel._interpolate(self.cached_CDF, x, *indices)

    def _pdf(self, x, *args):
        if not isinstance(x, collections.Iterable):
            return self._do_pdf(x)
        else:
            return [self._do_pdf(i) for i in x]

    def _do_pdf(self, x):
        # TODO(Max): improve performance with multiprocessing
        try:
            indices = Pandel._find_index(x, self.xs)
        except IndexError:
            # means that x is too small/big to fall into the calculated range → PDF at x is very close to 0
            return 0
        return Pandel._interpolate(self.cached_PDF, x, *indices)

    @staticmethod
    def _find_index(true_x, available_xs):
        # find index
        index = available_xs.searchsorted(true_x)
        # PDF should evaluate to 0 anyways if true_x is so small/big → return any valid index
        if index == 0 or index == available_xs.size:
            raise IndexError()
        lower, higher = available_xs[index - 1], available_xs[index]
        assert lower <= true_x <= higher
        return index, lower, higher

    @staticmethod
    def _interpolate(PDF_values, true_x, index, lower_value, higher_value):
        width = higher_value - lower_value
        pos = higher_value - true_x
        interpolation_factor = pos / width
        assert 0 <= interpolation_factor <= 1, interpolation_factor
        return float(
            PDF_values[index - 1] * (1 - interpolation_factor)
            + PDF_values[index] * interpolation_factor
        )

