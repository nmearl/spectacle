import operator

import astropy.units as u
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import _FitterMeta, LevMarLSQFitter
from astropy.modeling.models import Const1D, RedshiftScaleFactor
from astropy.convolution import Kernel1D
from specutils import Spectrum1D

from .converters import FluxConvert, FluxDecrementConvert
from .profiles import OpticalDepth1D
from .lsfs import COSLSFModel, GaussianLSFModel, LSFModel
from ..fitting.curve_fitter import CurveFitter
from ..utils.misc import DOPPLER_CONVERT

__all__ = ['Spectral1D']


class DynamicFittable1DModelMeta(type):
    """
    Meta class that acts as a factory for creating compound models with
    variable collections of sub models. This can take a collection of
    :class:`~spectacle.modeling.profiles.OpticalDepth1D` and return a new
    compound model.
    """
    def __call__(cls, lines=None, continuum=None, z=0, lsf=None, output='flux',
                 velocity_convention='relativistic', **kwargs):
        # If no continuum is provided, or the continuum provided is not a
        # model, use a constant model to represent the continuum.
        if continuum is not None and isinstance(continuum, Fittable1DModel):
            if isinstance(continuum, (float, int)):
                continuum = Const1D(amplitude=continuum,
                                    fixed={'amplitude': True})
        elif isinstance(continuum, (float, int)):
            continuum = Const1D(amplitude=continuum, fixed={'amplitude': True})
        else:
            continuum = Const1D(amplitude=0, fixed={'amplitude': True})

        if output not in ('flux', 'flux_decrement', 'optical_depth'):
            raise ValueError("Parameter 'output' must be one of 'flux', "
                             "'flux_decrement', 'optical_depth'.")

        # Parse the lines argument which can be a list, a quantity, or a string
        _lines = []

        if isinstance(lines, Fittable1DModel):
            _lines.append(lines)
        elif isinstance(lines, str):
            _lines.append(OpticalDepth1D(name=lines))
        elif isinstance(lines, list):
            for line in lines:
                if isinstance(line, str):
                    _lines.append(OpticalDepth1D(name=line))
                elif isinstance(line, u.Quantity):
                    _lines.append(OpticalDepth1D(lambda_0=line))
                elif isinstance(line, Fittable1DModel):
                    _lines.append(line)

        # Parse the lsf information, if provided
        if lsf is not None and not isinstance(lsf, LSFModel):
            if isinstance(lsf, Kernel1D):
                lsf = LSFModel(kernel=lsf)
            elif isinstance(lsf, str):
                if lsf == 'cos':
                    lsf = COSLSFModel()
                elif lsf == 'gaussian':
                    lsf = GaussianLSFModel(kwargs.pop('stddev', 1))
            else:
                raise ValueError("Kernel must be of type 'LSFModel', or "
                                 "'Kernel1D'; or a string with value 'cos' "
                                 "or 'gaussian'.")

        # Compose the line-based compound model taking into consideration
        # the redshift, continuum, and dispersion conversions.
        rs = RedshiftScaleFactor(z, fixed={'z': True}).inverse

        if lines is not None:
            ln = np.sum(_lines)

            if output == 'flux_decrement':
                compound_model = (rs | (continuum + (ln | FluxDecrementConvert())) | rs.inverse)
            elif output == 'flux':
                compound_model = (rs | (continuum + (ln | FluxConvert())) | rs.inverse)
            else:
                compound_model = (rs | (continuum + ln) | rs.inverse)
        else:
            compound_model = (rs | continuum | rs.inverse)

        # Check for any lsf kernels that have been added
        if lsf is not None:
            compound_model |= lsf

        # Add the definitions inside the original Spectral1D class to the new
        # compound class. Creating a new class seems to be less error-prone
        # than dynamically changing the base of a pre-existing class.
        _cls_kwargs = {}
        _cls_kwargs.update(cls.__dict__)
        _cls_kwargs['continuum'] = continuum
        _cls_kwargs['_velocity_convention'] = velocity_convention

        Spectral1D = type("Spectral1D", (compound_model.__class__,), _cls_kwargs)

        # Override the call function on the returned generated compound model
        # class in order to return a Spectrum1D object.
        # _set_custom_call(Spectral1D)

        # Updating the dict directly on an instance instead of during the type
        # creation above avoids issues of python complaining that the __dict__
        # object does not belong to the class.
        # spec1d.__dict__.update(cls.__dict__)

        return type.__call__(Spectral1D, **kwargs)


class Spectral1D(metaclass=DynamicFittable1DModelMeta):
    """
    Base representation of a compound model containing a variable number of
    :class:`~spectacle.modeling.profiles.OpticalDepth1D` line model features.

    Parameters
    ----------
    lines : str, :class:`~OpticalDepth1D`, list
        The line information used to compose the spectral model. This can be
        either a string, in which case the line information is retrieve from the
        ion registry; an instance of :class:`~OpticalDepth1D`; or a list of
        either of the previous two types.
    continuum : :class:`~Fittable1DModel`, optional
        An astropy model representing the continuum of the spectrum. If not
        provided, a :class:`~Const1D` model is used.
    z : float, optional
        The redshift applied to the spectral model. Default = 0.
    lsf : :class:`~spectacle.modeling.lsfs.LSFModel`, :class:`~astropy.convolution.Kernel1D`, str, optional
        The line spread function applied to the spectral model. It can be a
        pre-defined kernel model, or a convolution kernel, or a string
        referencing the built-in Hubble COS lsf, or a Gaussian lsf. Optional
        keyword arguments can be passed through.
    """
    inputs = ('x',)
    outputs = ('y',)
    input_units_allow_dimensionless = False

    def fit_to(self, x, y, fitter=CurveFitter(), rest_wavelength=None, **kwargs):
        # if not fitter.__class__ in _FitterMeta.registry:
        #     raise Exception("Fitter must be an astropy fitter subclass.")

        # The internal models assume all inputs are in km/s, if provided a
        # quantity object, ensure that it is converted to the proper units.
        if isinstance(x, u.Quantity):
            if x.unit.physical_type == 'speed':
                if rest_wavelength is None:
                    raise ValueError("A `rest_wavelength` must be included when the"
                                     "dispersion is provided in velocity space.")

                disp_equiv = u.spectral() + DOPPLER_CONVERT[
                    self._velocity_convention](rest_wavelength)

                with u.set_enabled_equivalencies(disp_equiv):
                    x = x.to('Angstrom')
            elif x.unit.physical_type in ('length', 'frequency'):
                x = x.to('Angstrom', equivalencies=u.spectral())

        # A data array returned by a Spectrum1D object is implicitly a Quantity
        # and should be reverted to a regular array for the unitless fitter.
        if isinstance(y, u.Quantity):
            y = y.value

        # Create a new compound without units that can be used with the
        # astropy fitters, since compound models with units are not
        # currently supported.
        unitless_cls, parameter_units = _strip_units(self, x)

        fitted_model = fitter(unitless_cls(), x.value, y, **kwargs)

        # Now we put back the units on the model
        model_with_units = _apply_units(fitted_model, parameter_units)

        # Attach any estimates parameter errors to the new fitted model
        if hasattr(fitter, 'errors'):
            model_with_units.errors = fitter.errors
        else:
            model_with_units.errors = None

        return model_with_units()

    @property
    def redshift(self):
        """
        The redshift at which the data given to.

        Returns
        -------
        : float
            The redshift value.
        """
        return next((x for x in self
                     if isinstance(x, RedshiftScaleFactor))).inverse.z.value

    @property
    def lines(self):
        """
        The collection of profiles representing the absorption or emission
        lines in the spectrum model.

        Returns
        -------
        : list
            A list of :class:`~spectacle.modeling.profiles.Voigt1D` models.
        """
        return [x for x in self if isinstance(x, OpticalDepth1D)]

    @property
    def lsf_kernel(self):
        return next((x for x in self if isinstance(x, LSFModel)), None)

    @property
    def output_type(self):
        """
        The data output of this spectral model. It could one of 'flux',
        'flux_decrement', or 'optical_depth'.

        Returns
        -------
        : str
            The output type of the model.
        """
        for x in self:
            if isinstance(x, FluxConvert):
                return 'flux'
            elif isinstance(x, FluxDecrementConvert):
                return 'flux_decrement'
        else:
            return 'optical_depth'

    def _copy(self, **kwargs):
        """
        Copy the spectral model, optionally overriding any previous values.

        Parameters
        ----------
        kwargs : dict
            A dictionary holding the values desired to be overwritten.

        Returns
        -------
        : :class:`~spectacle.modeling.models.Spectral1D`
            The new spectral model.
        """
        new_kwargs = dict(
            lines=self.lines,
            continuum=self.continuum,
            z=self.redshift,
            output=self.output_type,
            lsf=self.lsf_kernel)

        new_kwargs.update(kwargs)

        return Spectral1D(**new_kwargs)

    @property
    def as_flux(self):
        """New spectral model that produces flux output."""
        return self._copy(output='flux')

    @property
    def as_flux_decrement(self):
        """New spectral model that produces flux decrement output."""
        return self._copy(output='flux_decrement')

    @property
    def as_optical_depth(self):
        """New spectral model that produces optical depth output."""
        return self._copy(output='optical_depth')

    def with_lsf(self, kernel=None, **kwargs):
        """New spectral model with a line spread function."""
        return self._copy(lsf=kernel, **kwargs)

    def with_line(self, *args, **kwargs):
        """
        Add a new line to the spectral model.

        Returns
        -------
        : :class:`~spectacle.modeling.models.Spectral1D`
            The new spectral model.
        """
        new_line = OpticalDepth1D(*args, **kwargs)

        return self._copy(lines=self.lines + [new_line])


class ProfileMultiplier(Fittable1DModel):
    inputs = ('x',)
    outputs = ('y',)

    count = Parameter(default=0, min=0, tied=lambda x: int(x))

    @staticmethod
    def evaluate(self, count, *args, **kwargs):


        return self._copy()


def _set_custom_call(cls):
    def _custom_call(self, x, *args, **kwargs):
        data = super(cls, self).__call__(x, *args, **kwargs)
        return Spectrum1D(flux=u.Quantity(data), spectral_axis=u.Quantity(x))

    setattr(cls, '__call__', _custom_call)


# Model arithmetic operators
OPERATORS = {'+': operator.add,
             '-': operator.sub,
             '*': operator.mul,
             '/': operator.truediv,
             '**': operator.pow,
             '&': operator.and_,
             '|': operator.or_}


def _strip_units(compound_model, x=None):
    """
    Remove the units of a given compound model.

    Parameters
    ----------
    compound_model : :class:`~Fittable1D`
        Compound model for which the units will be removed.
    x : :class:`~astropy.units.Quantity`, optional
        The dispersion array that will be passed into the compound model. If
        provided, relevant parameters of the compound model will be converted
        to the unit.

    Returns
    -------
    : :class:`~Fittable1D`
        Compound model without units.
    """
    leaf_idx = -1

    parameter_units = {pn: getattr(sm, pn).unit
                       for sm in compound_model for pn in sm.param_names}

    def getter(idx, model):
        # By indexing on self[] this will return an instance of the
        # model, with all the appropriate parameters set
        sub_mod = compound_model[idx]

        for pn in sub_mod.param_names:
            param = getattr(sub_mod, pn)

            if param.unit is not None:
                # if x is not None and isinstance(x, u.Quantity):
                #     print("Value", sub_mod.lambda_0.quantity)
                #     with u.set_enabled_equivalencies(
                #             u.spectral() + u.doppler_relativistic(lambda_0)):
                #         quant = param.quantity.to(x.unit)
                # else:
                quant = param.quantity

                param.value = quant.value
                param._set_unit(None, force=True)

        return sub_mod

    unitless_model = compound_model._tree.evaluate(OPERATORS, getter=getter).__class__
    unitless_model._parameter_units = parameter_units

    # Set custom call to return a Spectrum1D object
    # _set_custom_call(unitless_model)

    return unitless_model, parameter_units


def _apply_units(compound_model, parameter_units):
    """
    Applies a set of units to a compound model.

    Parameters
    ----------
    compound_model : :class:`~Fittable1D`
        Unitless compound model.
    parameter_units : dict
        Dictionary containing a mapping between parameter names in the sub
        models of the compound model and the units to be associated with them.

    Returns
    -------
    : :class:`~Fittable1D`
        Compound model with units.
    """
    leaf_idx = -1

    def getter(idx, model):
        # By indexing on self[] this will return an instance of the
        # model, with all the appropriate parameters set
        sub_mod = compound_model[idx]

        for pn in sub_mod.param_names:
            param = getattr(sub_mod, pn)

            unit = parameter_units.get(pn)
            param._set_unit(unit, force=True)

        return sub_mod

    unitful_model = compound_model._tree.evaluate(OPERATORS, getter=getter).__class__

    # Set custom call to return a Spectrum1D object
    # _set_custom_call(unitful_model)

    return unitful_model