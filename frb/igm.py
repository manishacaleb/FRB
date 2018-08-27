""" Module for IGM calculations
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import pdb

from pkg_resources import resource_filename

from scipy.interpolate import interp1d

from astropy import units
from astropy.table import Table
from astropy.utils import isiterable
from astropy.cosmology import Planck15
from astropy import constants

def fukugita04_dict():
    """
    Data from Fukugita 2004, Table 1

    Returns
    -------
    f04_dict: dict

    """
    f04_dict = {}
    f04_dict['M_sphere'] = 0.0015
    f04_dict['M_disk'] = 0.00055
    f04_dict['M_HI'] = 0.00062
    f04_dict['M_H2'] = 0.00016
    f04_dict['M_WD'] = 0.00036
    f04_dict['M_NS'] = 0.00005
    f04_dict['M_BH'] = 0.00007
    f04_dict['M_BD'] = 0.00014
    # Return
    return f04_dict


def average_fHI(z, z_reion=7.):
    """
    Average HI fraction
    1 = neutral
    0 = fully ionized


    Parameters
    ----------
    z
    z_reion

    Returns
    -------
    fHI: float or ndarray
    """
    z, flg_z = z_to_array(z)
    fHI = np.zeros_like(z)
    #
    zion = z > z_reion
    fHI[zion] = 1.
    # Return
    if flg_z:
        return fHI
    else:
        return fHI[0]


def average_He_nume(z, z_HIreion=7., z_HeIIreion=3.):
    """
    Average number of electrons contributed by He as a function of redshift
    per He nucleus


    Parameters
    ----------
    z: float or ndarray
      Redshift

    Returns
    -------
    neHe: ndarray
      Number of free electrons per Helium nucelus

    """
    z, flg_z = z_to_array(z)
    fHeI = np.zeros_like(z)
    fHeII = np.zeros_like(z)
    # HeI ionized at HI reionization
    zion = z > z_HIreion
    fHeI[zion] = 1.
    # HeII ionized at HeII reionization
    zion2 = z > z_HeIIreion
    fHeII[zion2] = 1.
    # Combine
    neHe = (1.-fHeI) + (1.-fHeII)  #  No 2 on the second term as the first one gives you the first electron
    # Return
    if flg_z:
        return neHe
    else:
        return neHe[0]

def average_DM(z, cosmo=None, cumul=False, neval=10000, mu=1.3):
    """
    Calculate the average DM 'expected' based on our empirical
    knowledge of baryon distributions and their ionization state.

    Parameters
    ----------
    z: float
      Redshift
    mu: float
      Reduced mass correction for He when calculating n_H
    cumul: bool, optional
      Return the DM as a function of z

    Returns
    -------
    DM: Quantity
      One value if cumul=False
      else evaluated at a series of z (neval)
    zeval: ndarray, optional
      Evaluation redshifts if cumul=True

    """
    # Cosmology
    if cosmo is None:
        cosmo = Planck15
    # Init
    zeval = np.linspace(0., z, neval)[1:]

    # Get baryon mass density
    rho_b = cosmo.Ob0 * cosmo.critical_density0.to('Msun/Mpc**3') * (1+zeval)**3

    # Dense components
    rho_Mstar = avg_rhoMstar(zeval, remnants=True)
    rho_ISM = avg_rhoISM(zeval)

    # Diffuse
    rho_diffuse = rho_b - (rho_Mstar+rho_ISM)

    # Here we go
    n_H = (rho_diffuse/constants.m_p/mu).to('cm**-3')
    n_He = n_H / 12.  # 25% He mass fraction

    n_e = n_H * (1.-average_fHI(zeval)) + n_He*(average_He_nume(zeval))

    # Cosmology -- 2nd term is the (1+z) factor for DM
    denom = np.sqrt((1+zeval)**3 * cosmo.Om0 + cosmo.Ode0) * (1+zeval) * (1+zeval)

    # Time to Sum
    dz = zeval[1] - zeval[0]
    DM_cum = ((constants.c/cosmo.H0) * np.cumsum(n_e * dz / denom)).to('pc/cm**3')

    # Return
    if cumul:
        return DM_cum, zeval
    else:
        return DM_cum[-1]


def avg_rhoISM(z):
    """
    Co-moving Mass density of the ISM

    Assumes z=0 properties for z<1
    and otherwise M_ISM = M* for z>1

    Parameters
    ----------
    z: float or ndarray
      Redshift

    Returns
    -------
    rhoISM : Quantity
      Units of Msun/Mpc^3

    """
    # Init
    z, flg_z = z_to_array(z)
    rhoISM_unitless = np.zeros_like(z)
    # Mstar
    rhoMstar = avg_rhoMstar(z, remnants=False)
    # z<1
    f04_dict = fukugita04_dict()
    M_ISM = f04_dict['M_HI'] + f04_dict['M_H2']
    f_ISM = M_ISM/(f04_dict['M_sphere']+f04_dict['M_disk'])
    lowz = z<1
    rhoISM_unitless[lowz] = f_ISM * rhoMstar[lowz].value
    # z>1
    rhoISM_unitless[~lowz] = rhoMstar[~lowz].value
    # Finish
    rhoISM = rhoISM_unitless * units.Msun / units.Mpc**3
    #
    return rhoISM



def avg_rhoMstar(z, remnants=True):
    """
    Return mass density in stars as calculated by
    Madau & Dickinson (2014)

    Parameters
    ----------
    z: float or ndarray
      Redshift
    remnants: bool, optional
      Include remnants in the calculation?

    Returns
    -------
    rho_Mstar: Quantity
      Units of Msun/Mpc^3

    """
    # Init
    z, flg_z = z_to_array(z)
    # Load
    stellar_mass_file = resource_filename('frb', 'data/IGM/stellarmass.dat')
    rho_mstar_tbl = Table.read(stellar_mass_file, format='ascii')
    # Output
    rho_Mstar_unitless = np.zeros_like(z)

    # Extrema
    highz = z > rho_mstar_tbl['z'][-1]
    rho_Mstar_unitless[highz] = rho_mstar_tbl['rho_Mstar'][-1]

    # Interpolate
    fint = interp1d(rho_mstar_tbl['z'], rho_mstar_tbl['rho_Mstar'], kind='cubic')
    rho_Mstar_unitless[~highz] = fint(z[~highz])

    # Finish
    rho_Mstar = rho_Mstar_unitless * units.Msun / units.Mpc**3

    # Remnants
    if remnants:
        # Fukugita 2004 (Table 1)
        f04_dict = fukugita04_dict()
        f_remnants = (f04_dict['M_WD'] + f04_dict['M_NS'] + f04_dict['M_BH'] + f04_dict['M_BD']) / (
                f04_dict['M_sphere'] + f04_dict['M_disk'])
        # Apply
        rho_Mstar *= (1+f_remnants)

    # Return
    if flg_z:
        return rho_Mstar
    else:
        return rho_Mstar[0]


def avg_rhoSFR(z):
    """
    Average SFR density

    Based on Madau & Dickinson (2014)

    Parameters
    ----------
    z: float or ndarray
      Redshift

    Returns
    -------
    SFR: Quantity
      Units of Msun/yr/Mpc^3

    """
    rho_SFR_unitless = 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
    rho_SFR = rho_SFR_unitless * units.Msun / units.yr / units.Mpc**3

    # Return
    return rho_SFR

def z_to_array(z):
    """
    Convert input scalar or array to an array

    Parameters
    ----------
    z: float or ndarray
      Redshift

    Returns
    -------
    z: ndarray
    flg_z: int
      0 -- Input was a scalar
      1 -- Input was an array

    """
    # float or ndarray?
    if not isiterable(z):
        z = np.array([z])
        flg_z = 0
    else:
        flg_z = 1
    # Return
    return z, flg_z