import logging

import astropy.units as u
import numpy as np
from astropy.constants import c, m_e, e
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import RedshiftScaleFactor

from .curve_fitter import CurveFitter
from ..modeling import OpticalDepth1D, Spectral1D
from ..utils.detection import region_bounds, profile_type
from ..utils.misc import DOPPLER_CONVERT
from ..registries import line_registry

TAU_FACTOR = (np.sqrt(np.pi) * e.esu ** 2 /
              (m_e.cgs * c.cgs)).cgs


class LineFinder1D(Fittable2DModel):
    """
    The line finder class used to discover ion profiles within spectral data.

    Parameters
    ----------
    ions : list
        The list of ions to consider when discovering centroids. Each found
        centroid will be assigned an ion from this list, or the entire
        ion database if no filter is provided.
    continuum : float, :class:`~astropy.modeling.fitting.Fittable1DModel`
        Either a value representing the continuum's constant value, or an
        astropy fittable model representing the continuum. Used in fitting and
        added to the final spectral model produced by the operation.
    defaults : dict
        Dictionary containing key-value pairs when the key is the parameter
        name accepted by the :class:`~Spectral1D` class. If a parameter is
        defined this way, the fitter will use it instead of the fitted
        parameter value.
    z : float
        The redshift applied to the spectral model. Default = 0.
    output : {'optical_depth', 'flux', 'flux_decrement'}
        The expected output when evaluating the model object. Default =
        'optical_depth'.
    velocity_convention : {'optical', 'radio', 'relativistic'}
        The velocity convention to use when converting between wavelength and
        velocity space dispersion values. Default = 'relativistic'.
    """
    inputs = ('x', 'y')
    outputs = ('y',)

    @property
    def input_units_allow_dimensionless(self):
        return {'x': False, 'y': True}

    threshold = Parameter(default=0, fixed=True)
    min_distance = Parameter(default=10.0, min=1, fixed=True)

    def __init__(self, ions=None, continuum=None, defaults=None, z=None,
                 auto_fit=True, output='flux',
                 velocity_convention='relativistic', fitter=None,
                 auto_reject=False, fitter_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ions = ions or []
        self._continuum = continuum
        self._defaults = defaults or {}
        self._redshift_model = RedshiftScaleFactor(z=z or 0)
        self._model_result = None
        self._auto_fit = auto_fit
        self._output = output
        self._velocity_convention = velocity_convention
        self._fitter_args = fitter_args or {}
        self._fitter = fitter or CurveFitter()
        self._auto_reject = auto_reject

    @property
    def model_result(self):
        return self._model_result

    @property
    def fitter(self):
        return self._fitter

    def __call__(self, x, *args, auto_fit=None, **kwargs):
        if auto_fit is not None:
            self._auto_fit = auto_fit

        if x.unit.physical_type == 'speed' and len(self._ions) != 1:
            raise ReferenceError("The line finder will not be able to parse "
                                 "ion information in velocity space without "
                                 "being given explicit ion reference in the "
                                 "defaults dictionary.")

        super().__call__(x, *args, **kwargs)

        return self._model_result

    def evaluate(self, x, y, threshold, min_distance, *args, **kwargs):
        spec_mod = Spectral1D(continuum=self._continuum, output=self._output)

        if x.unit.physical_type in ('length', 'frequency'):
            x = self._redshift_model.inverse(x)

        # Generate the subset of the table for the ions chosen by the user
        sub_registry = line_registry

        if len(self._ions) > 0:
            # In this case, the user has provided a list of ions for their
            # spectrum. Create a subset of the line registry so that only
            # these ions will be searched when attempting to identify.
            sub_registry = line_registry.subset(self._ions)

        # Convert the min_distance from dispersion units to data elements.
        # Assumes uniform spacing.
        # min_ind = (np.abs(x.value - (x[0].value + min_distance))).argmin()

        # Find peaks
        regions = region_bounds(x, y, threshold=threshold,
                                min_distance=min_distance)

        # First, generate profiles based on all the primary peaks
        prime_lines = self._discover_lines(
            x, y, spec_mod, sub_registry,
            {k: v for k, v in regions.items() if not v[3]},
            threshold, min_distance)

        prime_mod = Spectral1D(prime_lines,
                               continuum=0,
                               output='optical_depth',
                               z=self._redshift_model.z.value)

        # Second, mask out the input data based on the found primary peaks,
        # then fit the buried lines.
        py = prime_mod(x)

        thresh_mask = np.greater(py/np.max(py), 0.0001)
        px, py = x[~thresh_mask], y[~thresh_mask]
        my = np.interp(x, px, py)

        tern_lines = self._discover_lines(
            x, my, spec_mod, sub_registry,
            {k: v for k, v in regions.items() if v[3]},
            threshold, min_distance)

        # Generate the final spectral model
        spec_mod = Spectral1D(prime_lines + tern_lines,
                              continuum=self._continuum,
                              output=self._output,
                              z=self._redshift_model.z.value)

        if self._auto_fit:
            if issubclass(self._fitter.__class__, LevMarLSQFitter):
                if 'maxiter' not in self._fitter_args:
                    self._fitter_args['maxiter'] = 1000

            if self._auto_reject:
                fit_spec_mod, _, _, _ = spec_mod.rejection_criteria(
                    self._redshift_model(x), y, auto_fit=True,
                    fitter=self._fitter, fitter_args=self._fitter_args)
            else:
                fit_spec_mod = self._fitter(spec_mod, self._redshift_model(x),
                                            y, **self._fitter_args)
        else:
            if self._auto_reject:
                fit_spec_mod, _, _, _ = spec_mod.rejection_criteria(
                    self._redshift_model(x), y, auto_fit=False)
            else:
                fit_spec_mod = spec_mod

        # The parameter values on the underlying compound model must also be
        # updated given the new fitted parameters on the Spectral1D instance.
        # FIXME: when fitting without using line finder, these values will not
        # be updated in the compound model.
        for pn in fit_spec_mod.param_names:
            pv = getattr(fit_spec_mod, pn)
            setattr(fit_spec_mod._compound_model, pn, pv)

        fit_spec_mod.line_regions = regions

        self._model_result = fit_spec_mod

        return fit_spec_mod(x)

    def _discover_lines(self, x, y, mod, registry, regions, threshold,
                        min_distance):
        lines = []

        # First, calculate all the primary line profiles
        for centroid, (mn_bnd, mx_bnd), is_absorption, buried in regions.values():
            mn_bnd, mx_bnd = mn_bnd * x.unit, mx_bnd * x.unit
            sub_x, vel_mn_bnd, vel_mx_bnd = None, None, None

            line_kwargs = {}

            # For the case where the user has provided a list of ions with a
            # dispersion in wavelength or frequency, convert each ion to
            # velocity space individually to avoid making assumptions of their
            # kinematics.
            if x.unit.physical_type in ('length', 'frequency'):
                line = registry.with_lambda(centroid)

                disp_equiv = u.spectral() + DOPPLER_CONVERT[
                    self._velocity_convention](line['wave'])

                with u.set_enabled_equivalencies(disp_equiv):
                    sub_x = u.Quantity(x, 'km/s')
                    vel_mn_bnd, vel_mx_bnd, vel_centroid = mn_bnd.to('km/s'), \
                                                           mx_bnd.to('km/s'), \
                                                           centroid.to('km/s')
            else:
                line = registry.with_name(self._ions[0])

            line_kwargs.update({
                'name': line['name'],
                'lambda_0': line['wave'],
                'gamma': line['gamma'],
                'f_value': line['osc_str']})

            # Estimate the doppler b and column densities for this line.
            # For the parameter estimator to be accurate, the spectrum must be
            # continuum subtracted.
            centroid, v_dop, col_dens, nmn_bnd, nmx_bnd = parameter_estimator(
                centroid=centroid,
                bounds=(vel_mn_bnd or mn_bnd, vel_mx_bnd or mx_bnd),
                x=sub_x or x,
                y=mod.continuum(sub_x or x) - y if is_absorption else y,
                ion_info=line_kwargs,
                buried=buried)

            if np.isinf(col_dens) or np.isnan(col_dens):
                continue

            estimate_kwargs = {
                'v_doppler': v_dop,
                'column_density': col_dens,
                'fixed': {},
                'bounds': {},
            }

            # Depending on the dispersion unit information, decide whether
            # the fitter should consider delta values in velocity or
            # wavelength/frequency space.
            if x.unit.physical_type in ('length', 'frequency'):
                estimate_kwargs['delta_lambda'] = centroid - line['wave']
                estimate_kwargs['fixed'].update({'delta_v': True})
                estimate_kwargs['bounds'].update({
                    'delta_lambda': (mn_bnd.value - line['wave'].value,
                                     mx_bnd.value - line['wave'].value)})
            else:
                # In velocity space, the centroid *should* be zero for any
                # line given that the rest wavelength is taken as its lamba_0
                # in conversions. Thus, the given centroid is akin to the
                # velocity offset.
                estimate_kwargs['delta_v'] = centroid
                estimate_kwargs['fixed'].update({'delta_lambda': True})
                estimate_kwargs['bounds'].update({
                    'delta_v': (mn_bnd.value, mx_bnd.value)})

            line_kwargs.update(estimate_kwargs)
            line_kwargs.update(self._defaults.copy())

            line = OpticalDepth1D(**line_kwargs)
            lines.append(line)

        logging.debug(
            "Found %s possible lines (theshold=%s, min_distance=%s).",
            len(lines), threshold, min_distance)

        return lines


def parameter_estimator(centroid, bounds, x, y, ion_info, buried=False):
    # bound_low, bound_up = bounds
    # mid_diff = (bound_up - bound_low)
    # new_bound_low, new_bound_up = (bound_low - mid_diff), (bound_up + mid_diff)

    new_bound_low, new_bound_up = bounds
    mask = ((x >= new_bound_low) & (x <= new_bound_up))
    mx, my = x[mask], y[mask]

    # X centroid estimates the position
    centroid = np.sum(mx * my) / np.sum(my)

    # width can be estimated by the weighted
    # 2nd moment of the X coordinate.
    dx = mx - np.mean(mx)
    fwhm = 2 * np.sqrt(np.sum((dx * dx) * my) / np.sum(my))
    sigma = fwhm / 2.355
    # sigma = sigma * 2 if buried else sigma

    # import matplotlib.pyplot as plt

    # f, ax = plt.subplots()

    # ax.plot(x, y)
    # ax.plot(mx, my)

    # amplitude is derived from area.
    height = np.abs(np.max(y)) - np.abs(np.min(y))

    # Estimate the doppler b parameter
    v_dop = (np.sqrt(2) * sigma).to('km/s')

    # Estimate the column density
    col_dens = (height * v_dop / (TAU_FACTOR * ion_info['lambda_0'] *
                                  ion_info['f_value'])).to('1/cm2')
    # col_dens = (sum_y / (TAU_FACTOR * ion_info['lambda_0'] * ion_info['f_value'])).to('1/cm2')
    ln_col_dens = np.log10(col_dens.value)

    # if buried:
    #     ln_col_dens -= 0.1

    logging.info("""Estimated initial values:
    Ion: {}
    Centroid: {:g} ({:g})
    Column density: {:g}, ({:g})
    Doppler width: {:g}""".format(ion_info['name'], centroid,
                                  ion_info['lambda_0'], ln_col_dens,
                                  col_dens, v_dop))

    return centroid, v_dop, ln_col_dens, new_bound_low, new_bound_up
