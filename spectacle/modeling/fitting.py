import peakutils
from scipy.signal import savgol_filter
from astropy.modeling import fitting, models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table
from astropy.extern import six
import numpy as np
import abc
import six
from scipy import optimize

from ..core.utils import ION_TABLE, find_index
from ..core.spectra import Spectrum1D
from ..core.models import Spectrum1DModel


@six.add_metaclass(abc.ABCMeta)
class Fitter:
    def __init__(self, noise=3.5, min_dist=100):
        self.detrend = False
        self.noise = noise
        self.min_dist = min_dist
        self.fit_info = None

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _find_lines(self, spectrum, strict=False):
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

        indexes = peakutils.indexes(
            inv_flux,
            thres=np.std(inv_flux) * self.noise/np.max(inv_flux),
            min_dist=self.min_dist)

        indexes = np.array(indexes)

        if strict:
            # Find the indices of the ion table that correspond with the found
            # indices in the peak search
            tab_indexes = np.array(list(set(
                map(lambda x: find_index(ION_TABLE['wave'], x),
                    spectrum.dispersion[indexes]))))

            # Given the indices in the ion tab, find the indices in the
            #  dispersion array that correspond to the ion table lambda
            indexes = np.array(list(
                map(lambda x: find_index(spectrum.dispersion, x),
                    ION_TABLE['wave'][tab_indexes])))

        return indexes

    def _find_continuum(self, spectrum, mode='LinearLSQFitter'):
        data_med = np.median(spectrum.data)
        cont = models.Linear1D(slope=0.0, intercept=data_med)

        fitter = getattr(fitting, mode)()

        # We only want to use weights if there is an appreciable difference
        # between the median and the continuum
        diff = np.abs(data_med - spectrum.data)
        weights = diff ** -3 if np.sum(diff) > np.min(spectrum.data) else None

        cont_fit = fitter(cont, spectrum.dispersion, spectrum.data,
                          weights=weights)

        return cont_fit


class LevMarFitter(Fitter):
    def __call__(self, spectrum, model=None, strict=False):
        if model is None:
            # Create a new spectrum object with the same dispersion
            model = Spectrum1DModel()

            # Find the lines in the flux array
            indexes = self._find_lines(spectrum, strict=strict)

            print("found {} lines".format(len(indexes)))

            # Add a new line to the empty spectrum object for each found line
            for ind in indexes:
                model.add_line(lambda_0=spectrum.dispersion[ind],
                               v_doppler=1e7, column_density=10**14,
                               # name=ION_TABLE['name'][ind]
                               )

        # Create fitter instance
        # fitter = LevMarCurveFitFitter()
        fitter = LevMarLSQFitter()

        model_fit = fitter(model.model,
                           spectrum.dispersion,
                           spectrum.data,
                           # sigma=spectrum.uncertainty.array,
                           maxiter=min(1000, 250 * len(model._line_models))
                           )

        # Grab the covariance matrix
        param_cov = fitter.fit_info['param_cov']

        if param_cov is None:
            param_cov = np.zeros(len(model_fit.param_names))

        # This is not robust. The covariance matrix does not include
        # constrained parameters, so we must insert some value for the
        # variance to make sure the number of parameters match.
        # for i, val in enumerate(model_fit.param_names):
        #     if model_fit.tied.get(val) or model_fit.fixed.get(val):
        #         param_cov = np.insert(param_cov, i, 0.0)

        # Update fit info
        self.fit_info = Table([model_fit.param_names,
                               model.model.parameters,
                               model_fit.parameters,
                               param_cov],
                              names=("Parameter", "Original Value",
                                     "Fitted Value", "Uncertainty"))

        # Update model with new parameters
        model.model.parameters = model_fit.parameters

        return model(spectrum.dispersion)


@six.add_metaclass(fitting._FitterMeta)
class LevMarCurveFitFitter(object):
    """
    Levenberg-Marquardt algorithm and least squares statistic based on
    SciPy's `curve_fit` in order to allow passing in of data errors.

    Attributes
    ----------
    covar : 2d array
        The estimated covariance of the optimal parameter values. The
        diagonals provide the variance of the parameter estimate. See
        `scipy.optimize.curve_fit` documentation for more details.

    """

    # The constraint types supported by this fitter type.
    supported_constraints = ['fixed', 'tied', 'bounds']

    def __init__(self):
        self.fit_info = {'nfev': None,
                         'fvec': None,
                         'fjac': None,
                         'ipvt': None,
                         'qtf': None,
                         'message': None,
                         'ierr': None,
                         'param_jac': None,
                         'param_cov': None}

        super(LevMarCurveFitFitter, self).__init__()

    def objective_wrapper(self, model):
        def objective_function(dispersion, *param_vals):
            """
            Function to minimize.
            Parameters
            ----------
            fps : list
                parameters returned by the fitter
            args : list
                [model, [weights], [input coordinates]]
            """
            fitting._fitter_to_model_params(model, param_vals)

            return np.ravel(model(dispersion))

        return objective_function

    def __call__(self, model, x, y, z=None, sigma=None,
                 maxiter=fitting.DEFAULT_MAXITER, acc=fitting.DEFAULT_ACC,
                 epsilon=fitting.DEFAULT_EPS, estimate_jacobian=False,
                 **kwargs):
        """
        Fit data to this model.

        Returns
        -------
        model_copy : `~astropy.modeling.FittableModel`
            a copy of the input model with parameters set by the fitter
        """
        model_copy = fitting._validate_model(model, self.supported_constraints)

        if model_copy.fit_deriv is None or estimate_jacobian:
            dfunc = None
        else:
            dfunc = self._wrap_deriv

        init_values, _ = fitting._model_to_fit_params(model_copy)

        fit_params, cov_x = optimize.curve_fit(
            self.objective_wrapper(model_copy), x, y,
            # method='trf',
            p0=init_values,
            sigma=sigma,
            epsfcn=epsilon,
            maxfev=maxiter,
            col_deriv=model_copy.col_fit_deriv,
            xtol=acc,
            absolute_sigma=True,
            **kwargs)

        fitting._fitter_to_model_params(model_copy, fit_params)

        cov_diag = []
        for i in range(len(fit_params)):
            try:
                cov_diag.append(np.absolute(cov_x[i][i]) ** 0.5)
            except:
                cov_diag.append(0.00)

        self.fit_info['cov_x'] = cov_x
        self.fit_info['param_cov'] = cov_diag

        return model_copy

    @staticmethod
    def _wrap_deriv(params, model, weights, x, y, z=None):
        """
        Wraps the method calculating the Jacobian of the function to account
        for model constraints.
        `scipy.optimize.leastsq` expects the function derivative to have the
        above signature (parlist, (argtuple)). In order to accommodate model
        constraints, instead of using p directly, we set the parameter list in
        this function.
        """

        if weights is None:
            weights = 1.0

        if any(model.fixed.values()) or any(model.tied.values()):

            if z is None:
                full_deriv = np.ravel(weights) * np.array(
                    model.fit_deriv(x, *model.parameters))
            else:
                full_deriv = (np.ravel(weights) * np.array(
                    model.fit_deriv(x, y, *model.parameters)).T).T

            pars = [getattr(model, name) for name in
                    model.param_names]
            fixed = [par.fixed for par in pars]
            tied = [par.tied for par in pars]
            tied = list(
                np.where([par.tied is not False for par in pars],
                         True, tied))
            fix_and_tie = np.logical_or(fixed, tied)
            ind = np.logical_not(fix_and_tie)

            if not model.col_fit_deriv:
                full_deriv = np.asarray(full_deriv).T
                residues = np.asarray(
                    full_deriv[np.nonzero(ind)]).T
            else:
                residues = full_deriv[np.nonzero(ind)]

            return [np.ravel(_) for _ in residues]
        else:
            if z is None:
                return [np.ravel(_) for _ in
                        np.ravel(weights) * np.array(
                            model.fit_deriv(x, *params))]
            else:
                return [np.ravel(_) for _ in (
                np.ravel(weights) * np.array(model.fit_deriv(x, y, *params)).T).T]
