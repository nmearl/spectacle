from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Linear1D
from astropy.table import Table
import numpy as np
import os

from .utils import find_index, ION_TABLE
from .profiles import TauProfile
from .spectra import Spectrum1D


class Voigt1D(Fittable1DModel):
    """
      Implements a Voigt profile (convolution of Cauchy-Lorentz
      and Gaussian distribution).
    """
    lambda_0 = Parameter()
    f_value = Parameter(min=0, max=1.0)
    gamma = Parameter(min=0)
    v_doppler = Parameter()
    column_density = Parameter(min=1e10, max=1e30)
    delta_v = Parameter(default=0)
    delta_lambda = Parameter(default=0)

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        #lambda_bins = self.meta.get('lambda_bins', None)
        profile = TauProfile(lambda_0=lambda_0, f_value=f_value,
                             gamma=gamma, v_doppler=v_doppler,
                             column_density=column_density,
                             n_lambda=x.size,# lambda_bins=lambda_bins,
                             delta_v=delta_v, delta_lambda=delta_lambda)

        # if lambda_bins is None:
        #     self.meta['lambda_bins'] = profile.lambda_bins

        flux = np.exp(-profile.optical_depth) - 1.0

        return flux

    @staticmethod
    def fit_deriv(x, x_0, b, gamma, f):
        return [0, 0, 0, 0]

    # @property
    # def lambda_bins(self):
    #     return self.meta.get('lambda_bins')


class Spectrum1DModel:
    def __init__(self):
        self._continuum_model = None
        self._line_models = []
        self._model = None

    def __repr__(self):
        return self.model.__repr__()

    @property
    def model(self):
        """
        Returns the complete :class:`astropy.modeling.Models` object.
        """
        if self._continuum_model is None:
            self._continuum_model = Linear1D(slope=0.0, intercept=1.0)

        if self._model is None or self._line_models != len(self._model):
            model = self._continuum_model

            if len(self._line_models) > 0:
                model = model + np.sum(self._line_models)

            self._model = model

        return self._model

    def __call__(self, dispersion):
        """
        Applies the compound model to the dispersion axis to produce the
        idealized version of the spectrum.

        Returns
        -------
        data : ndarray
            The data result from the compound model.
        """
        data = self.model(dispersion)

        mod_spec = Spectrum1D(data, dispersion=dispersion)

        return mod_spec

    @property
    def continuum(self, dispersion):
        return self._continuum_model(dispersion)

    @property
    def line_list(self):
        """
        List all available line names.
        """
        return ION_TABLE

    def add_line(self, v_doppler, column_density, lambda_0=None, f_value=None,
                 gamma=None, delta_v=None, delta_lambda=None, name=None):
        if name is not None:
            ind = np.where(ION_TABLE['name'] == name)
            lambda_0 = ION_TABLE['wave'][ind]
        else:
            ind = find_index(ION_TABLE['wave'], lambda_0)
            name = ION_TABLE['name'][ind]

        if f_value is None:
            f_value = ION_TABLE['osc_str'][ind]

        model = Voigt1D(lambda_0=lambda_0, f_value=f_value, gamma=gamma or 0,
                        v_doppler=v_doppler, column_density=column_density,
                        delta_v=delta_v, delta_lambda=delta_lambda, name=name,
                        #meta={'lambda_bins': self.dispersion}
                        )

        # If gamma has not been explicitly defined, tie it to lambda
        if gamma is None:
            gamma_val = ION_TABLE['gamma'][ind]
            model.gamma.value = gamma_val
            model.gamma.tied = lambda cmod, mod=model: _tie_gamma(cmod, mod)

        self._line_models.append(model)

        # Force the compound model to be recreated
        self._model = None

        return model

    def remove_line(self, model=None, x_0=None):
        if model is not None:
            self._line_models.remove(model)
        elif x_0 is not None:
            model = self.get_profile(x_0)
            self._line_models.remove(model)

    def get_profile(self, x_0):
        # Find the nearest voigt profile to the given central wavelength
        v_arr = sorted(self._line_models, key=lambda x: x.lambda_0.value)
        v_x_0_arr = np.array([x.lambda_0.value for x in v_arr])

        if len(v_x_0_arr) > 1:
            ind = find_index(v_x_0_arr, x_0)

            # Retrieve the voigt profile at that wavelength
            v_prof = v_arr[ind]
        else:
            v_prof = v_arr[0]

        return v_prof

    def _get_range_mask(self, x_0=None):
        profile = np.sum(self._line_models) #self.get_profile(x_0 or 0.0)
        vdisp = profile(self.dispersion)
        cont = np.zeros(self.dispersion.shape)

        return ~np.isclose(vdisp, cont, rtol=1e-2, atol=1e-5)


def _tie_gamma(compound_model, model):
    # Find the index of the original model in the compound model
    mod_ind = compound_model._submodels.index(model)

    # The auto-generated name of the parameter in the compound model
    param_name = "lambda_0_{}".format(mod_ind)
    lambda_val = getattr(compound_model, param_name).value

    ind = find_index(ION_TABLE['wave'], lambda_val)
    gamma_val = ION_TABLE['gamma'][ind]

    return gamma_val