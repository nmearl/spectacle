from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Linear1D
from astropy.table import Table
import numpy as np
import os
import logging
from scipy.signal import savgol_filter
import peakutils

from ..core.utils import find_nearest
from ..core.profiles import TauProfile
from ..core.spectra import Spectrum1D
from ..core.registries import line_registry


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
    delta_v = Parameter(default=0, fixed=True)
    delta_lambda = Parameter(default=0)

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        #lambda_bins = self.meta.get('lambda_bins', None)
        profile = TauProfile(x, lambda_0=lambda_0, f_value=f_value,
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


class Absorption1D:
    def __init__(self, line_list=None):
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

        if self._model is None or len(self._line_models) != len(self._model):
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

        lines = {l.name: l.lambda_0.value for l in self._line_models}
        mod_spec = Spectrum1D(data, dispersion=dispersion, lines=lines)

        return mod_spec

    def copy(self, from_model=None):
        new_model = self.__class__()

        if from_model is not None:
            new_model._continuum_model = from_model._submodels[0]
            new_model._line_models = from_model._submodels[1:]

        return new_model

    @property
    def continuum(self, dispersion):
        return self._continuum_model(dispersion)

    @property
    def parameters(self):
        return self.model.parameters

    def set_parameters_from_model(self, model):
        self._continuum_model.parameters = model[0].parameters

        for i in range(1, len(model.submodel_names)):
            self._line_models[i-1].parameters = model[i].parameters

    def add_line(self, v_doppler, column_density, lambda_0=None, f_value=None,
                 gamma=None, delta_v=None, delta_lambda=None, name=None):
        if name is not None:
            ind = np.where(line_registry['name'] == name)[0]

            if len(ind) == 0:
                logging.error("No line with name {} found.".format(name))
                return
            elif len(ind) > 1:
                logging.warning(
                    "Multiple lines found with name {}.".format(name))

            ind = ind[0]

            if lambda_0 is None:
                lambda_0 = line_registry['wave'][ind]
        elif lambda_0 is not None:
            ind = find_nearest(line_registry['wave'], lambda_0)
            name = line_registry['name'][ind]
        else:
            logging.error("Unable to estimate line values from given "
                          "information.")
            return

        if f_value is None:
            f_value = line_registry['osc_str'][ind]

        model = Voigt1D(lambda_0=lambda_0, f_value=f_value, gamma=gamma or 0,
                        v_doppler=v_doppler, column_density=column_density,
                        delta_v=delta_v, delta_lambda=delta_lambda, name=name)

        # If gamma has not been explicitly defined, tie it to lambda
        if gamma is None:
            gamma_val = line_registry['gamma'][ind]
            model.gamma.value = gamma_val
            model.gamma.tied = lambda cmod, mod=model: _tie_gamma(cmod, mod)

        self._line_models.append(model)

        # Force the compound model to be recreated
        self._model = None

        return model

    def has_line(self, name=None, x_0=None):
        if name is not None:
            return self.get_line(name=name) is not None
        elif x_0 is not None:
            return self.get_line(x_0=x_0) is not None

    def remove_line(self, model=None, name=None, x_0=None):
        if model is not None:
            self._line_models.remove(model)
        elif name is not None:
            model = self.get_line(name=name)
            self._line_models.remove(model)
        elif x_0 is not None:
            model = self.get_profile(x_0)
            self._line_models.remove(model)

    def get_line(self, name=None, x_0=None):
        if name is not None:
            lines = [x for x in self._line_models if x.name == name]

            if len(lines) > 0:
                return lines[0]

        elif x_0 is not None:
            return self.get_profile(x_0)

    def get_profile(self, x_0):
        """
        Retrieves the specific Voigt profile from the model object.

        Parameters
        ----------
        x_0 : float
            The lambda value used to retrieve the related Voigt profile.

        Returns
        -------
        v_prof : :class:`spectacle.modeling.models.Voigt1D`
            The Voigt profile at the given lambda value.
        """
        # Find the nearest voigt profile to the given central wavelength
        v_arr = sorted(self._line_models, key=lambda x: x.lambda_0.value)
        v_x_0_arr = np.array([x.lambda_0.value for x in v_arr])

        if len(v_x_0_arr) > 1:
            ind = find_nearest(v_x_0_arr, x_0)

            # Retrieve the voigt profile at that wavelength
            v_prof = v_arr[ind]
        else:
            v_prof = v_arr[0]

        return v_prof

    def get_range_mask(self, dispersion, x_0=None):
        """
        Returns a mask for the model spectrum that indicates where the values
        are discernible from the continuum. I.e. where the absorption data
        is contained.

        If a `x_0` value is provided, only the related Voigt profile will be
        considered, otherwise, the entire model is considered.

        Parameters
        ----------
        dispersion : array-like
            The x values for which the model spectrum will be calculated.
        x_0 : float, optional
            The lambda value from which to grab a specific Voigt profile.

        Returns
        -------
        array-like
            Boolean array indicating indices of interest.

        """
        profile = self.model if x_0 is None else self.get_profile(x_0)
        vdisp = profile(dispersion)
        cont = np.zeros(dispersion.shape)

        return ~np.isclose(vdisp, cont, rtol=1e-2, atol=1e-5)

    def find_lines(self, spectrum, threshold=0.7, min_dist=100, strict=False):
        """
        Simple peak finder.

        Parameters
        ----------
        spectrum : :class:`spectacle.core.spectra.Spectrum1D`
            Original spectrum object.
        strict : bool
            If `True`, will require that any found peaks also exist in the
            provided ions lookup table.

        Returns
        -------
        indexes : np.array
            An array of indices providing the peak locations in the original
            spectrum object.
        """
        continuum = np.median(spectrum.data)
        inv_flux = 1 - spectrum.data

        # Filter with SG
        y = savgol_filter(inv_flux, 49, 3)

        indexes = peakutils.indexes(
            y,
            thres=threshold, # np.std(inv_flux) * self.noise/np.max(inv_flux),
            min_dist=min_dist)

        self._indices = indexes

        if strict:
            print("Using strict line associations.")
            # Find the indices of the ion table that correspond with the found
            # indices in the peak search
            tab_indexes = np.array(list(set(
                map(lambda x: find_nearest(line_registry['wave'], x),
                    spectrum.dispersion[indexes]))))

            # Given the indices in the ion tab, find the indices in the
            #  dispersion array that correspond to the ion table lambda
            indexes = np.array(list(
                map(lambda x: find_nearest(spectrum.dispersion, x),
                    line_registry['wave'][tab_indexes])))

        logging.info("Found {} lines.".format(len(indexes)))

        # Add a new line to the empty spectrum object for each found line
        for ind in indexes:
            il = find_nearest(line_registry['wave'],
                              spectrum.dispersion[ind])

            nearest_name = line_registry['name'][il]
            nearest_wave = line_registry['wave'][il]

            logging.info("Found {} ({}) at {}. Strict is {}.".format(
                nearest_name,
                nearest_wave,
                spectrum.dispersion[ind],
                strict))

            mod = self.add_line(lambda_0=spectrum.dispersion[ind],
                                 v_doppler=1e6, column_density=10**14,
                                 name=nearest_name,
                                 delta_v=0
                                 )

        logging.info("There are {} lines in the model.".format(
            len(self._line_models)))

        return indexes


def _tie_gamma(compound_model, model):
    # Find the index of the original model in the compound model
    mod_ind = compound_model._submodels.index(model)

    # The auto-generated name of the parameter in the compound model
    param_name = "lambda_0_{}".format(mod_ind)
    lambda_val = getattr(compound_model, param_name).value

    ind = find_nearest(line_registry['wave'], lambda_val)
    gamma_val = line_registry['gamma'][ind]

    return gamma_val