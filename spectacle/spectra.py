from .lsf import LSF
from .utils import find_index

import numpy as np
import numpy.ma as ma
import astropy.units as u
from astropy.convolution import convolve
import astropy.modeling.models as models

import logging


class Spectrum1D:
    def __init__(self, dispersion=None, flux=None):
        self._flux = flux
        self._dispersion = dispersion
        self._mask = None
        self._lsfs = []
        self._line_models = []
        self._continuum_model = None
        self._remat = None
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._continuum_model + np.sum(self._line_models)

        return self._model

    @property
    def dispersion(self):
        dispersion = self._dispersion

        if dispersion is None:
            dispersion = np.arange(0, 5000)

        if self._remat is not None:
            dispersion = np.dot(self._remat, self._dispersion)

        return ma.masked_array(dispersion, self._mask)

    @property
    def flux(self):
        if self._flux is None:
            flux = self.model(self._dispersion)
        else:
            flux = self._flux

        # Apply LSFs
        for lsf in self._lsfs:
            flux = convolve(flux, lsf.kernel)

        # Apply resampling
        if self._remat is not None:
            flux = np.dot(self._remat, flux)

        flux = ma.masked_array(flux, self._mask)

        return flux

    @property
    def ideal_flux(self):
        flux = self.model(self._dispersion)

        return flux

    @property
    def continuum(self):
        return self._continuum_model(self.dispersion)

    def copy(self):
        spectrum_copy = self.__class__()
        spectrum_copy._flux = self._flux
        spectrum_copy._dispersion = self._dispersion
        spectrum_copy._mask = self._mask
        spectrum_copy._lsfs = self._lsfs
        spectrum_copy._line_models = self._line_models
        spectrum_copy._continuum_model = self._continuum_model
        spectrum_copy._remat = self._remat
        spectrum_copy._model = self._model

        return spectrum_copy

    def set_mask(self, mask):
        if np.array(mask).shape != self._flux.shape or \
                        np.array(mask).shape != self._dispersion.shape:
            logging.warning("Mask shape does not match data shape.")
            return

        self._mask = mask

    def add_lsf(self, function='gaussian', *args, **kwargs):
        lsf = LSF(function, *args, **kwargs)
        self._lsfs.append(lsf)

    def add_line(self, amp, mu, gamma, sigma=None, normalize=False, name=""):
        model = models.Voigt1D(x_0=mu, amplitude_L=amp, fwhm_L=gamma,
                               fwhm_G=sigma or gamma)
        self._line_models.append(model)

        # Force the compound model to be recreated
        self._model = None

        return model

    def set_continuum(self, function, *args, **kwargs):
        model = getattr(models, function)
        self._continuum_model = model(*args, **kwargs)

    def fwhm(self, x_0):
        """
          Calculates an approximation of the FWHM.

          The approximation is accurate to
          about 0.03% (see http://en.wikipedia.org/wiki/Voigt_profile).

          Returns
          -------
          FWHM : float
              The estimate of the FWHM
        """
        # Find the nearest voigt profile to the given central wavelength
        v_arr = sorted(self._line_models, key=lambda x: x.x_0)
        v_x_0_arr = [x.x_0 for x in v_arr]
        ind = find_index(v_x_0_arr, x_0)

        # Retrive the voigt profile at that wavelength
        v_prof = v_arr[ind]

        # The width of the Lorentz profile
        fl = 2.0 * v_prof.gamma

        # Width of the Gaussian [2.35 = 2*sigma*sqrt(2*ln(2))]
        fd = 2.35482 * v_prof.sigma

        return 0.5346 * fl + np.sqrt(0.2166 * (fl ** 2.) + fd ** 2.)

    def resample(self, dispersion):
            remat = self._resample_matrix(self.dispersion, dispersion)
            self._remat = remat

    def _resample_matrix(self, orig_lamb, fin_lamb):
        """
        Create a resampling matrix to be used in resampling spectra in a way
        that conserves flux. This is adapted from code created by the SEAGal
        Group.

        .. note:: This method assumes uniform grids.

        Parameters
        ----------
        orig_lamb : ndarray
            The original dispersion array.
        fin_lamb : ndarray
            The desired dispersion array.

        Returns
        -------
        resample_map : ndarray
            An [[N_{fin_lamb}, M_{orig_lamb}]] matrix.
        """
        # Get step size
        delta_orig = orig_lamb[1] - orig_lamb[0]
        delta_fin = fin_lamb[1] - fin_lamb[0]

        n_orig_lamb = len(orig_lamb)
        n_fin_lamb = len(fin_lamb)

        # Lower bin and upper bin edges
        orig_low = orig_lamb - delta_orig * 0.5
        orig_upp = orig_lamb + delta_orig * 0.5
        fin_low = fin_lamb - delta_fin * 0.5
        fin_upp = fin_lamb + delta_fin * 0.5

        # Create resampling matrix
        resamp_mat = np.zeros(shape=(n_fin_lamb, n_orig_lamb))

        for i in range(n_fin_lamb):
            # Calculate the contribution of each original bin to the
            # resampled bin
            l_inf = np.where(orig_low > fin_low[i], orig_low, fin_low[i])
            l_sup = np.where(orig_upp < fin_upp[i], orig_upp, fin_upp[i])

            # Interval overlap of each original bin for current resampled
            # bin; negatives clipped
            dl = (l_sup - l_inf).clip(0)

            # This will only happen at the edges of lorig.
            # Discard resampled bin if it's not fully covered (> 99%) by the
            #  original bin -- only happens at the edges of the original bins
            if 0 < dl.sum() < 0.99 * delta_fin:
                dl = 0 * orig_lamb

            resamp_mat[i, :] = dl

        resamp_mat /= delta_fin

        return resamp_mat
