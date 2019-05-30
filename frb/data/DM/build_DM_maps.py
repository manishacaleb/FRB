""" Module for buiding DM maps
"""

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys, json
import warnings
from IPython import embed

import healpy as hp
from frb.halos import LMC, SMC, M33
from frb.halos import ModifiedNFW, MB04, YF17, M31, ICM


#mpl.rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
#]

from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

import multiprocessing as mp

from astropy.cosmology import Planck15 as cosmo
from astropy import units

from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits

from ne2001 import density
from ne2001.density import NEobject

from frb import halos as frb_halos

def calc_ne_s(s, g=3):
    global n0, rc, rs, a, b, e, Rperp
    # r
    r = np.sqrt(Rperp ** 2 + s ** 2)
    #
    npne = n0 ** 2 * (r/rc)**(-a) / (1 + (r / rc)**2)**(3*b - a/2) / (1 + (r/rs)**g)**(e/g)
    val = np.sqrt(npne)
    return val

def load_sightlines():
    # Sightlines
    heal_pix = np.load('./hp_glon_glat.npy')  # Not sure how this was made!
    lons = heal_pix[0,:]
    lats = heal_pix[1,:]
    return lons, lats

# Generate DMs
ne = density.ElectronDensity()#**PARAMS)
def wrap_ne(iargs):
    return ne.DM(iargs[0], iargs[1], 100.).value


def build_ISM(hp_file='hp_DM_ISM.fits', ncpu=15):

    from ne2001 import density
    import time

    time0 = time.time()

    # Various models
    n_eval = 12 * 128**2
    nside = hp.npix2nside(n_eval)
    lons, lats = hp.pix2ang(nside, np.arange(n_eval), lonlat=True)

    coords = SkyCoord(l=lons*units.deg, b=lats*units.deg, frame='galactic')
    print("Number of calculations = {}".format(len(coords)))


    ls = coords.l.value
    bs = coords.l.value
    inps = []
    for kk in range(len(ls)):
        inps.append((ls[kk], bs[kk]))
    #embed(header='83 of build_DM')

    # Parallel
    pool = mp.Pool(ncpu)

    DMs = pool.map(wrap_ne, inps)
    #for jj,coord in enumerate(coords):
    #    #DMs[jj] = ne.DM(coord.l.value, coord.b.value, 100., epsabs=1e-4, epsrel=1e-4).value
    #    DMs[jj] = ne.DM(coord.l.value, coord.b.value, 100.).value
    #    #DMs[jj] = ne.DM(coord.l.value, coord.b.value, 100., integrator=np.sum).value
    #    if (jj%100) == 0:
    #        print(jj, coord, DMs[jj])

    # Write
    t = Table()
    t['flux'] = DMs  # the data array
    t.meta['ORDERING'] = 'RING'
    t.meta['COORDSYS'] = 'G'
    t.meta['NSIDE'] = nside
    t.meta['INDXSCHM'] = 'IMPLICIT'
    t.write(hp_file, overwrite=True)
    time1 = time.time()

    print("This took {}s".format(time1-time0))



def build_LG(hp_file='hp_DM_LG.fits', Rmin=10.):
    """ Build DM for the Local Group, including our Galaxy"""
    nside = 1024
    lons, lats = hp.pix2ang(nside, np.arange(12* nside**2), lonlat=True)

    # M31
    m31 = M31(alpha=2, y0=2)
    M31_b = m31.coord.galactic.b.value
    M31_l = m31.coord.galactic.l.value

    # LMC, SMC, M33
    lmc = LMC()
    smc = SMC()
    m33 = M33()

    # MW
    warnings.warn("DO THIS RIGHT SOMEDAY!")
    '''
    Mhalo = np.log10(1.5e12) # Boylan-Kolchin et al. 2013
    f_hot = 0.75  # Allows for disk + ISM
    c = 7.7
    mnfw_2 = ModifiedNFW(log_Mhalo=Mhalo, f_hot=f_hot, y0=2, alpha=2, c=c)
    # Zero out inner 10kpc
    mnfw_2.zero_inner_ne = 10.  # kpc
    params = dict(F=1., e_density=1.)
    model_ne = NEobject(mnfw_2.ne, **params)
    DM_MW = np.zeros_like(lons)
    for kk in range(lons.size):
        DM_MW[kk] = model_ne.DM(lons[kk], lats[kk], mnfw_2.r200.value).value
        if (kk%100) == 0:
            print(kk, DM_MW[kk])
    '''
    MW_DM = 60.*np.ones(lons.size)

    # Halo
    Rvals = np.linspace(Rmin/10., m31.r200.value, 200) # For splining
    DMvals = np.zeros_like(Rvals)
    LMC_vals = np.zeros_like(Rvals)
    SMC_vals = np.zeros_like(Rvals)
    M33_vals = np.zeros_like(Rvals)
    for kk, iRperp in enumerate(Rvals):
        DMvals[kk] = m31.Ne_Rperp(iRperp * units.kpc).to('pc/cm**3').value
        LMC_vals[kk] = lmc.Ne_Rperp(iRperp * units.kpc).to('pc/cm**3').value
        SMC_vals[kk] = smc.Ne_Rperp(iRperp * units.kpc).to('pc/cm**3').value
        M33_vals[kk] = smc.Ne_Rperp(iRperp * units.kpc).to('pc/cm**3').value
    fDM = interp1d(Rvals, DMvals, kind='cubic')
    LMC_fDM = interp1d(Rvals, LMC_vals, kind='cubic')
    SMC_fDM = interp1d(Rvals, SMC_vals, kind='cubic')
    M33_fDM = interp1d(Rvals, M33_vals, kind='cubic')

    # Do the calculation

    coords = SkyCoord(l=lons*units.deg, b=lats*units.deg, frame='galactic')

    # Rperp
    #Set origin at Sun(0, 0)
    #Set center of M31 at d_M31 along x - axis(d_M31, 0)
    #Line: ax + by + c = 0
    #Slope: m = -a / b = tan(theta) with theta the angle off of M31
    #Let a = 1
    a = 1
    c = 0
    x0, y0 = m31.distance.to('kpc').value, 0. # kpc (Riess, A.G., Fliri, J., & Valls - Gabaud, D. 2012, ApJ, 745, 156)

    sep = m31.coord.separation(coords)
    atan = np.arctan(sep.radian)
    b = -1 * a / atan
    Rperp = np.abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)

    M31_DM = np.zeros_like(Rperp)

    in_ism = Rperp <= Rmin
    in_halo = (Rperp < m31.r200.value) & (Rperp > Rmin)
    M31_DM[in_ism] = np.max(DMvals)
    M31_DM[in_halo] = fDM(Rperp[in_halo])

    # Add in the Clouds
    lmc_x0, lmc_y0 = lmc.distance.to('kpc').value, 0.
    sep = lmc.coord.separation(coords)
    atan = np.arctan(sep.radian)
    b = -1 * a / atan
    lmc_Rperp = np.abs(a * lmc_x0 + b * lmc_y0 + c) / np.sqrt(a ** 2 + b ** 2)
    in_lmc = (lmc_Rperp < lmc.r200.value) & (lmc_Rperp > Rmin/10.)
    lmc_ism = lmc_Rperp < Rmin/10.
    LMC_DM = np.zeros_like(Rperp)
    LMC_DM[in_lmc] = LMC_fDM(lmc_Rperp[in_lmc])
    LMC_DM[lmc_ism] = LMC_vals[0]

    # SMC
    smc_x0, smc_y0 = smc.distance.to('kpc').value, 0.
    sep = smc.coord.separation(coords)
    atan = np.arctan(sep.radian)
    b = -1 * a / atan
    smc_Rperp = np.abs(a * smc_x0 + b * smc_y0 + c) / np.sqrt(a ** 2 + b ** 2)
    in_smc = (smc_Rperp < smc.r200.value) & (smc_Rperp > Rmin/10.)
    smc_ism = smc_Rperp < Rmin/10.
    SMC_DM = np.zeros_like(Rperp)
    SMC_DM[in_smc] = SMC_fDM(smc_Rperp[in_smc])
    SMC_DM[smc_ism] = SMC_vals[0]

    # M33
    m33_x0, m33_y0 = m33.distance.to('kpc').value, 0.
    sep = m33.coord.separation(coords)
    atan = np.arctan(sep.radian)
    b = -1 * a / atan
    m33_Rperp = np.abs(a * m33_x0 + b * m33_y0 + c) / np.sqrt(a ** 2 + b ** 2)
    in_m33 = (m33_Rperp < m33.r200.value) & (m33_Rperp > Rmin/10.)
    m33_ism = m33_Rperp < Rmin/10.
    M33_DM = np.zeros_like(Rperp)
    M33_DM[in_m33] = M33_fDM(m33_Rperp[in_m33])
    M33_DM[m33_ism] = M33_vals[0]

    DM_tot = M31_DM + LMC_DM + SMC_DM + M33_DM + MW_DM

    # Write
    t = Table()
    t['flux'] = DM_tot  # the data array
    t.meta['ORDERING'] = 'RING'
    t.meta['COORDSYS'] = 'G'
    t.meta['NSIDE'] = 1024
    t.meta['INDXSCHM'] = 'IMPLICIT'
    t.write(hp_file, overwrite=True)
    print("Wrote: {}".format(hp_file))



def build_LGM(hp_file='DM_LGM.fits'):

    Mhalo = 12.5 # Gives 10^13 Msun at 3Mpc (see LGM Notebeook)
    f_hot = 0.85  # Increased
    c = 7.7

    # Various models
    nfw = ModifiedNFW()
    mnfw_2 = ModifiedNFW(log_Mhalo=Mhalo, f_hot=f_hot, y0=2, alpha=2, c=c)

    # Sightlines
    heal_pix = np.load('../Figures/hp_glon_glat.npy')
    lons = heal_pix[0,:]
    lats = heal_pix[1,:]

    coords = SkyCoord(l=lons*units.deg, b=lats*units.deg, frame='galactic')
    rmax = 3300. # kpc -- 3 Mpc + a bit more for our position

    # Set up the model (see Notebook)
    params = dict(F=1., e_density=1.)
    model_ne = NEobject(mnfw_2.ne, **params)
    m31 = M31()
    d = m31.distance/2.
    Z = d * np.sin(-1 * m31.coord.galactic.b)
    X = np.sqrt((d ** 2 - Z ** 2) / (1 + np.tan(m31.coord.galactic.l + 90*units.deg) ** 2))
    Y = np.sqrt(d ** 2 - X ** 2 - Z ** 2)
    density.set_xyz_sun(np.array([-X.value, -Y.value, Z.value]))  # kpc

    # Generate DMs
    DMs = np.zeros(len(coords))
    for jj,coord in enumerate(coords):
        DMs[jj] = model_ne.DM(coord.l.value, coord.b.value, rmax).value
        if (jj%100) == 0:
            print(jj, coord, DMs[jj])

    # Write
    t = Table()
    t['flux'] = DMs  # the data array
    t.meta['ORDERING'] = 'RING'
    t.meta['COORDSYS'] = 'G'
    t.meta['NSIDE'] = 1024
    t.meta['INDXSCHM'] = 'IMPLICIT'
    t.write(hp_file, overwrite=True)


if __name__ == '__main__':

    flg_anly = 0
    if len(sys.argv) == 1:
        #flg_anly += 2**0   # Galactic ISM
        #flg_anly += 2**1   # LGM
        flg_anly += 2**2   # LG with our Galaxy

    if flg_anly & (2**0):
        build_ISM()

    if flg_anly & (2**1):
        build_LGM()

    if flg_anly & (2**2):
        build_LG()



