import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.io import fits
from scipy.interpolate import interp1d
from .utils import splitter, package_path
from .constants import ls_km
import pandas as pd


__all__ = ['StarSpectrum']



def get_star(star_type, velscale):
    '''
    Retrieves and processes the star data for the specified star type.

    Parameters
    ----------
    star_type : str
        The type of star ('A', 'F', 'G', 'K', or 'M').
    velscale: 
        The desired velocity scale for the re-binned spectrum.
    Returns
    -------
    tuple
        The processed star data: original wavelength, flux, rebinned wavelength, and normalized flux.
    '''
    star_list = {
        'A': 'HD_94601.txt',
        'F': 'HD_59881.txt',
        'G': 'HD_163917.txt',
        'K': 'HD_108381.txt',
        'M': 'HD_169305.txt'
    }
    
    # Read the spectrum data
    spec = pd.read_csv(f'{package_path}{splitter}data{splitter}{star_list[star_type]}', sep='\s+', comment='#', header=None, names=['lam', 'Flux'])
    
    lam = np.array(spec['lam'])
    flux = np.array(spec['Flux'])

    # Rebin the spectrum to the desired velocity scale
    flux_rebin, ln_lam_temp = log_rebin(lam, flux, velscale=velscale)[:2]
    lam_rebin = np.exp(ln_lam_temp)
    
    # Normalize the rebinned flux
    f_rebin_norm = flux_rebin / np.max(flux_rebin)
    
    return lam, flux, lam_rebin, f_rebin_norm

def log_rebin(lam, spec, velscale, oversample=1, flux=False):
    '''
    Logarithmically re-bins the spectrum to match the desired velocity scale.

    Parameters
    ----------
    lam : np.ndarray
        The wavelength array.
    spec : np.ndarray
        The flux or spectrum array.
    velscale : float
        The desired velocity scale for the re-binned spectrum. If None, it is calculated.
    oversample : int, optional
        Oversampling factor for the re-binning.
    flux : bool, optional
        If True, return flux. If False, return the wavelength.

    Returns
    -------
    tuple
        The rebinned spectrum, logarithmic wavelength array, and velocity scale.
    '''
    
    lam, spec = np.asarray(lam, dtype=float), np.asarray(spec, dtype=float)
    assert np.all(np.diff(lam) > 0), '`lam` must be monotonically increasing'
    n = len(spec)
    assert lam.size in [2, n], "`lam` must be either a 2-elements range or a vector with the length of `spec`"

    if lam.size == 2:
        dlam = np.diff(lam)/(n - 1)             # Assume constant dlam
        lim = lam + [-0.5, 0.5]*dlam
        borders = np.linspace(*lim, n + 1)
    else:
        lim = 1.5*lam[[0, -1]] - 0.5*lam[[1, -2]]
        borders = np.hstack([lim[0], (lam[1:] + lam[:-1])/2, lim[1]])
        dlam = np.diff(borders)

    ln_lim = np.log(lim)
    c = 299792.458                          # Speed of light in km/s

    if velscale is None:
        m = int(n*oversample)               # Number of output elements
        velscale = c*np.diff(ln_lim)/m      # Only for output (eq. 8 of Cappellari 2017, MNRAS)
        velscale = velscale.item()          # Make velscale a scalar
    else:
        ln_scale = velscale/c
        m = int(np.diff(ln_lim)/ln_scale)   # Number of output pixels

    newBorders = np.exp(ln_lim[0] + velscale/c*np.arange(m + 1))

    if lam.size == 2:
        k = ((newBorders - lim[0])/dlam).clip(0, n-1).astype(int)
    else:
        k = (np.searchsorted(borders, newBorders) - 1).clip(0, n-1)

    specNew = np.add.reduceat((spec.T*dlam).T, k)[:-1]    # Do analytic integral of step function
    specNew.T[...] *= np.diff(k) > 0                      # fix for design flaw of reduceat()
    specNew.T[...] += np.diff(((newBorders - borders[k]))*spec[k].T)    # Add to 1st dimension

    if not flux:
        specNew.T[...] /= np.diff(newBorders)   # Divide 1st dimension

    # Output np.log(wavelength): natural log of geometric mean
    ln_lam = 0.5*np.log(newBorders[1:]*newBorders[:-1])

    return specNew, ln_lam, velscale

class StarSpectrum(Fittable1DModel):
    """
    A class to represent a star spectrum model that can be fitted to data.

    Parameters
    ----------
    amplitude : float
        Amplitude of the star, units: arbitrary. Default is 1.
    sigma : float
        Velocity dispersion of the star, units: km/s. Default is 200.
    velscale : float
        The desired velocity scale for the rebinned spectrum. Default is 69.
    Star_type : str
        Type of the star ('A', 'F', 'G', 'K', 'M'). Default is 'A'.
    """
    
    amplitude = Parameter(default=1, bounds=(0, None))
    sigma = Parameter(default=200, bounds=(20, 6000))
    
    
    def __init__(self, amplitude=amplitude, sigma=sigma, velscale=69, Star_type='A', **kwargs):
        """
        Initializes the StarSpectrum model with the given parameters.

        Parameters
        ----------
        amplitude : float, optional
            Amplitude of the star, default is 1.
        sigma : float, optional
            Velocity dispersion, default is 200.
        velscale : float, optional
            Velocity scale for the rebinned spectrum, default is 69.
        star_type : str, optional
            Type of star to load data for, default is 'A'.
        """
        
        super().__init__(amplitude=amplitude, sigma=sigma, **kwargs)
        
        Star_x, Star_y, Star_x_rebin, Star_y_rebin_norm = get_star(Star_type, velscale)
        
        self.wave_temp = Star_x_rebin
        self.flux_temp = Star_y_rebin_norm
        
    def evaluate(self, x, amplitude, sigma):
        '''
        Stellar model function.
        '''
        
        if np.min(x) < self.wave_temp[0] or np.max(x) > self.wave_temp[-1]:
            raise ValueError(f'The wavelength is out of the supported range ({model_x_rebin[0]:.0f}-{model_x_rebin[-1]:.0f})!')
        
        s = sigma / ls_km
        
        ln_lam = np.log(self.wave_temp)
        
        nsig = s / (ln_lam[1] - ln_lam[0])
        nsig = max(nsig, 1e-6)
        flux_convolved = interp1d(self.wave_temp, gaussian_filter(self.flux_temp, nsig))(x)
        
        return amplitude * flux_convolved
        