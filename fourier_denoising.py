from bedmaster_processing.analysis.functional_interface.base_compute import Compute
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from bedmaster_processing.analysis import QueryAPI as Q
import bedmaster_processing.constants as CONST
import bedmaster_processing.utils as util
import bedmaster_processing.analysis.functional_interface.pan_tompkins_parallel_compute as ptpc
from bedmaster_processing.analysis import frequency_domain_analysis as fda
from scipy.signal import square
from scipy.integrate import simps as simpsons_integral
from scipy.integrate import trapz as trapezoid_integral
from scipy.optimize import fmin
from scipy.ndimage import label as binary_structuring
from scipy.stats import norm


class FourierDenoise(Compute):

    def __init__(self, debug):
        self.debug = debug

        self.f_x, self.f_y = None, None

    def DEBUG_ERRNO(msg, n):
        print('Error msg: ' + msg + ' Err #: {}'.format(n))

    def fourier_approx(self, ecg, coef_order=2):
        """
        Get coef_order coefficients for Fourier series approximation of ecg.
            Sinusoid coefficients are returned (a0, b0 are affine).
        :param ecg:
        :param coef_order:
        :return:
        """

        if len(ecg) == 0 and self.debug:
            DEBUG_ERRNO("No ECG provided", 0)

        try:
            ecg.std()
        except Exception:
            DEBUG_ERRNO("Non-numerical ECG", 1)

        ecg_approx = ecg.head(int(len(ecg) / config()['ECG']))

        # Fourier functions
        base_conv = lambda t: np.sin(t)
        yield_an = lambda n, E, x: 2.0 / len(E) * simpsons_integral(E * np.cos(2. * np.pi * n * x / len(E)))
        yield_bn = lambda n, E, x: 2.0 / len(E) * simpsons_integral(E * np.sin(2. * np.pi * n * x / len(E)))
        prev_an, prev_bn = [], []
        an, bn = [], []

        for i, r in ecg_approx.iterrows():
            sample = np.fromstring(r.ts[1:-1], dtype=int, sep=',')
            sample = sample[0:int(len(sample) / 4)]
            s0 = fmin(base_conv, np.array([0]), full_output=True, disp=False)
            s1 = fmin(base_conv, np.array([s0[0][0] + np.pi]), full_output=True, disp=False)
            conv_init = np.abs(s0[0][0])
            conv_ = np.abs(s1[0][0])
            s_x = np.linspace(conv_init, conv_, 25)
            s_x, s_y = s_x, base_conv(s_x)
            peaks = np.correlate(sample, s_y, mode="same") / np.max(np.correlate(sample, s_y, mode="same"))
            qrs_time = np.argwhere(peaks >= config()['QRS'])
            qrs_volt = sample[qrs_time].reshape(1, len(sample[qrs_time]))[0]

            T = 1. / len(qrs_volt)
            x = np.linspace(0, len(qrs_volt), len(qrs_volt), endpoint=False)
            a0 = T * 2. * simpsons_integral(qrs_volt, x)

            i = 0
            an, bn = [], []
            while i < coef_order - 1:
                an.append(yield_an(i, qrs_volt, x))
                bn.append(yield_bn(i, qrs_volt, x))
                i += 1

            if len(prev_an) == 0:
                prev_an = an
                prev_bn = bn
                continue

            print("New fourier coefficients: " + str(an) + "; " + str(bn))
            an_diff = np.abs(np.array(an) - np.array(prev_an)) / np.array(prev_an)
            bn_diff = np.abs(np.array(bn) - np.array(prev_bn)) / np.array(prev_bn)
            an_diff = an_diff <= config()['CONVERGENCE_MINIMUM']
            bn_diff = bn_diff <= config()['CONVERGENCE_MINIMUM']
            if np.all(np.concatenate((an_diff, bn_diff), axis=None)):
                print("Approximation converged. Exiting...")
                break

        def kernel(s):
            L = len(s)
            x = np.linspace(0, L, L, endpoint=False)
            ret = a0 / 2. + sum([yield_an(k, s, x) * np.cos(2. * np.pi * k * x / L) +
                                 yield_bn(k, s, x) * np.sin(2. * np.pi * k * x / L) for k in range(1, coef_order + 1)])
            return ret

        self.f_x = np.linspace(0, len(kernel(sample)), len(kernel(sample)), endpoint=False)
        self.f_y = kernel(sample)
        return sample, kernel(sample)

    def deprecated_conv_function(self, sample_ekg, coef_order=2):
        """
        First order example
        :param sample_ekg:
        :param coef_order:
        :return:
        """
        an, bn = self.fourier_approx(sample_ekg, coef_order)
        f = lambda t: bn[0] * np.cos(t) + an[0] * np.sin(t)

        x_0 = fmin(f, np.array([0]), full_output=True, disp=False)
        x_1 = fmin(f, np.array([x_0[0][0] + np.pi]), full_output=True, disp=False)

        conv_start = x_0[0][0]
        conv_end = x_1[0][0]
        f_x = np.linspace(conv_start, conv_end, 25)

        return f_x, f(f_x)

    def fourier_score(self, ekg, f_x, f_y):
        """
        :param ekg:
        :param f_x:
        :param f_y:
        :return:
        """
        norm_ekg = (ekg - ekg.mean()) / ekg.std()

        corr1d = np.correlate(norm_ekg, f_y, mode="same")
        integrable_score = corr1d / np.max(corr1d)
        qrs_complexes = np.argwhere(integrable_score >= config()['QRS'])
        qrs_voltages = ekg[qrs_complexes].reshape(1, len(ekg[qrs_complexes]))[0]
        diffs = np.diff(qrs_voltages) < config()['BINARY_STRUCTURE_DIFF']
        _, n = binary_structuring(diffs)

        # Could be atrial fibrillation
        if n == 0:
            n = 1e6

        score = trapezoid_integral(integrable_score ** 2)
        return score / n

    def score(self, row):
        """
        :param row:
        :return:
        """
        ekg = np.fromstring(row.ts[1:-1], dtype=int, sep=',')
        if len(ekg) < 240:
            return -1

        whole = self.fourier_score(ekg, self.f_x, self.f_y)
        tail = self.fourier_score(ekg[0: len(ekg) - config()['OFFSET']], self.f_x, self.f_y)
        head = self.fourier_score(ekg[config()['OFFSET']: len(ekg)], self.f_x, self.f_y)

        return min(head, min(whole, tail))

    def compute(self, row):
        return self.score(row)
