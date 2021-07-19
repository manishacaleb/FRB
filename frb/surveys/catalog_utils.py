""" Methods related to fussing with a catalog"""
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from astropy import units
from astropy.io import fits

from frb.galaxies.defs import valid_filters

import h5py, os

import warnings

from IPython import embed


def clean_heasarc(catalog):
    """
    Insure RA/DEC are ra/dec in the Table

    Table is modified in place

    Args:
        catalog (astropy.table.Table): Catalog generated by astroquery

    """
    # RA/DEC
    catalog.rename_column("RA", "ra")
    catalog.rename_column("DEC", "dec")
    for key in ['ra', 'dec']:
        catalog[key].unit = units.deg


def clean_cat(catalog, pdict, fill_mask=None):
    """
    Convert table column names intrinsic to the slurped
    catalog with the FRB survey desired values

    Args:
        catalog (astropy.table.Table): Catalog generated by astroquery
        pdict (dict):  Defines the original key and desired key
        fill_mask (int or float, optional):  Fill masked items with this value

    Returns:
        astropy.table.Table:  modified catalog

    """
    for key,value in pdict.items():
        if value in catalog.keys():
            catalog.rename_column(value, key)
    # Mask
    if fill_mask is not None:
        if catalog.mask is not None:
            catalog = catalog.filled(fill_mask)
    return catalog


def sort_by_separation(catalog, coord, radec=('ra','dec'), add_sep=True):
    """
    Sort an input catalog by separation from input coordinate

    Args:
        catalog (astropy.table.Table):  Table of sources
        coord (astropy.coordinates.SkyCoord): Reference coordinate for sorting
        radec (tuple): Defines catalog columns holding RA, DEC (in deg)
        add_sep (bool, optional): Add a 'separation' column with units of arcmin

    Returns:
        astropy.table.Table: Sorted catalog

    """
    # Check
    for key in radec:
        if key not in catalog.keys():
            print("RA/DEC key: {:s} not in your Table".format(key))
            raise IOError("Try again..")
    # Grab coords
    cat_coords = SkyCoord(ra=catalog[radec[0]].data,
                          dec=catalog[radec[1]].data, unit='deg')

    # Separations
    seps = coord.separation(cat_coords)
    isrt = np.argsort(seps)
    # Add?
    if add_sep:
        catalog['separation'] = seps.to('arcmin')
    # Sort
    srt_catalog = catalog[isrt]
    # Return
    return srt_catalog


def match_ids(IDs, match_IDs, require_in_match=True):
    """ Match input IDs to another array of IDs (usually in a table)
    Return the rows aligned with input IDs

    Args:
        IDs (ndarray): ID values to match
        match_IDs (ndarray):  ID values to match to
        require_in_match (bool, optional): Require that each of the
          input IDs occurs within the match_IDs

    Returns:
        ndarray: Rows in match_IDs that match to IDs, aligned -1 if there is no match

    """
    rows = -1 * np.ones_like(IDs).astype(int)
    # Find which IDs are in match_IDs
    in_match = np.in1d(IDs, match_IDs)
    if require_in_match:
        if np.sum(~in_match) > 0:
            raise IOError("qcat.match_ids: One or more input IDs not in match_IDs")
    rows[~in_match] = -1
    #
    IDs_inmatch = IDs[in_match]
    # Find indices of input IDs in meta table -- first instance in meta only!
    xsorted = np.argsort(match_IDs)
    ypos = np.searchsorted(match_IDs, IDs_inmatch, sorter=xsorted)
    indices = xsorted[ypos]
    rows[in_match] = indices
    return rows


def summarize_catalog(frbc, catalog, summary_radius, photom_column, magnitude):
    """
    Generate simple text describing the sources from
    an input catalog within a given radius

    Args:
        frbc: FRB Candidate object
        catalog (astropy.table.Table): Catalog table
        summary_radius (Angle):  Radius to summarize on
        photom_column (str): Column specifying which flux to work on
        magnitude (bool): Is the flux a magnitude?

    Returns:
        list: List of comments on the catalog

    """
    # Init
    summary_list = []
    coords = SkyCoord(ra=catalog['ra'], dec=catalog['dec'], unit='deg')
    # Find all within the summary radius
    seps = frbc['coord'].separation(coords)
    in_radius = seps < summary_radius
    # Start summarizing
    summary_list += ['{:s}: There are {:d} source(s) within {:0.1f} arcsec'.format(
        catalog.meta['survey'], np.sum(in_radius), summary_radius.to('arcsec').value)]
    # If any found
    if np.any(in_radius):
        # Brightest
        if magnitude:
            brightest = np.argmin(catalog[photom_column][in_radius])
        else:
            brightest = np.argmax(catalog[photom_column][in_radius])
        summary_list += ['{:s}: The brightest source has {:s} of {:0.2f}'.format(
            catalog.meta['survey'], photom_column,
            catalog[photom_column][in_radius][brightest])]
        # Closest
        closest = np.argmin(seps[in_radius])
        summary_list += ['{:s}: The closest source is at separation {:0.2f} arcsec and has {:s} of {:0.2f}'.format(
            catalog.meta['survey'],
            seps[in_radius][closest].to('arcsec').value,
            photom_column, catalog[photom_column][in_radius][brightest])]
    # Return
    return summary_list


def xmatch_catalogs(cat1, cat2, skydist = 5*units.arcsec,
                     RACol1 = "ra", DecCol1 = "dec",
                     RACol2 = "ra", DecCol2 = "dec"):
    """
    Cross matches two astronomical catalogs and returns
    the matched tables.
    Args:
        cat1, cat2: astropy Tables
            Two tables with sky coordinates to be
            matched.
        skydist: astropy Quantity, optional
            Maximum separation for a valid match.
            5 arcsec by default.
        RACol1, RACol2: str, optional
            Names of columns in cat1 and cat2
            respectively that contain RA in degrees.
        DecCol1, DecCol2: str, optional
            Names of columns in cat1 and cat2
            respectively that contain Dec in degrees.
        zCol1, zCol2: str, optional
            Names of columns in cat1 and cat2
            respectively that contain redshift in degrees.
            Matches in 3D if supplied. Both should be given.
    returns:
        match1, match2: astropy Table
            Tables of matched rows from cat1 and cat2.
    """

    # TODO add assertion statements to test input validity.
     
    # Get corodinates
    cat1_coord = SkyCoord(cat1[RACol1], cat1[DecCol1], unit = "deg")
    cat2_coord = SkyCoord(cat2[RACol2], cat2[DecCol2], unit = "deg")

    # Match 2D
    idx, d2d, _ = cat1_coord.match_to_catalog_sky(cat2_coord)

    # Get matched tables
    match1 = cat1[d2d < skydist]
    match2 = cat2[idx[d2d < skydist]]

    return match1, match2


def _detect_mag_cols(photometry_table):
    """
    Searches the column names of a 
    photometry table for columns with mags.
    Args:
        photometry_table: astropy Table
            A table containing photometric
            data from a catlog.
    Returns:
        mag_colnames: list
            A list of column names with magnitudes
        mag_err_colnames: list
            A list of column names with errors
            in the magnitudes.
    """
    assert type(photometry_table)==Table, "Photometry table must be an astropy Table instance."
    allcols = photometry_table.colnames
    photom_cols = np.array(valid_filters)
    photom_errcols = np.array([filt+"_err" for filt in photom_cols])

    photom_cols = photom_cols[[elem in allcols for elem in photom_cols]]
    photom_errcols = photom_errcols[[elem in allcols for elem in photom_errcols]]
    
    return photom_cols.tolist(), photom_errcols.tolist()


def mag_from_flux(flux, flux_err=None):
    """
    Get the AB magnitude from a flux

    Parameters
    ----------
    flux : Quantity
        Flux
    flux_err : Quantity
        Error in flux (optional)

    Returns
    -------
    mag, mag_err : float, float
        AB magnitude and its error (if flux_err is given)
        AB magnitude and `None` (if flux_err is `None`)
    """
    # convert flux to Jansky
    flux_Jy = flux.to('Jy').value

    # get mag
    mag_AB = -2.5*np.log10(flux_Jy) + 8.9

    # get error
    if flux_err is not None:
        flux_Jy_err = flux_err.to('Jy').value
        err_mag2 = (-2.5/np.log(10.) / flux_Jy)**2 * flux_Jy_err**2
        err_mag = np.sqrt(err_mag2)
    else:
        err_mag = None
    return mag_AB, err_mag

def _mags_to_flux(mag:float, zpt_flux:units.Quantity=3630.7805*units.Jy, mag_err:float=None)->float:
    """
    Convert a magnitude to mJy

    Args:
        mag (column): magnitude
        zpt_flux (Quantity, optional): Zero point flux for the magnitude.
            Assumes AB mags by default (i.e. zpt_flux = 3630.7805 Jy). 
        mag_err (float, optional): uncertainty in magnitude
    Returns:
        flux (float): flux in mJy
        flux_err (float): if mag_err is given, a corresponding
            flux_err is returned.
    """
    # Data validation
    #assert np.isreal(mag), "Mags must be floats."
    #assert (np.isreal(mag_err)) + (mag_err==None), "Mag errs must be floats"
    assert (type(zpt_flux) == units.Quantity)*(zpt_flux.decompose().unit == units.kg/units.s**2), "zpt_flux units should be Jy or with dimensions kg/s^2."

    flux = mag.copy()

    # Conver fluxes
    badmags = mag<-10
    flux[badmags] = -99.
    flux[~badmags] = zpt_flux.value*10**(-mag[~badmags]/2.5)
    
    if mag_err is not None:
        flux_err = mag_err.copy()
        baderrs = mag_err < 0
        flux_err[baderrs] = -99.
        flux_err[~baderrs] = flux[~baderrs]*(10**(mag_err[~baderrs]/2.5)-1)
        return flux, flux_err
    else:
        return flux    

def convert_mags_to_flux(photometry_table, fluxunits='mJy'):
    """
    Takes a table of photometric measurements
    in mags and converts it to flux units.

    Args:
        photometry_table (astropy.table.Table):
            A table containing photometric
            data from a catlog.
        fluxunits (str, optional):
            Flux units to convert the magnitudes
            to, as parsed by astropy.units. Default is mJy.
        Returns:
            fluxtable: astropy Table
                `photometry_table` but the magnitudes
                are converted to fluxes.
    """
    fluxtable = photometry_table.copy()
    mag_cols, mag_errcols = _detect_mag_cols(fluxtable)
    convert = units.Jy.to(fluxunits)
    #If there's a "W" in the column name, it's from WISE
    # TODO -- We need to deal with this hack
    wisecols = sorted([col for col in mag_cols if ("W" in col and 'WFC3' not in col)])
    wise_errcols = sorted([col for col in mag_errcols if ("W" in col and 'WFC3' not in col)])

    #Similarly define vista cols
    vistacols = sorted([col for col in mag_cols if "VISTA" in col])
    vista_errcols = sorted([col for col in mag_errcols if "VISTA" in col])

    fnu0 = {'WISE_W1':309.54,
            'WISE_W2':171.787,
            'WISE_W3':31.674,
            'WISE_W4':8.363,
            'VISTA_Y':2087.32,
            'VISTA_J':1554.03,
            'VISTA_H':1030.40,
            'VISTA_Ks':674.83} #http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux
                               #http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=Paranal&gname2=VISTA
    for mag,err in zip(wisecols+vistacols,wise_errcols+vista_errcols):
        flux, flux_err = _mags_to_flux(photometry_table[mag], fnu0[mag]*units.Jy, photometry_table[err])
        badflux = flux == -99.
        fluxtable[mag][badflux] = flux[badflux]
        fluxtable[mag][~badflux] = flux[~badflux]*convert
        #if flux != -99.:
        #    fluxtable[mag] = flux*convert
        #else:
        #    fluxtable[mag] = flux
        baderr = flux_err == -99.0
        fluxtable[err][baderr] = flux_err[baderr]
        fluxtable[err][~baderr] = flux_err[~baderr]*convert
        #if flux_err != -99.:
        #    fluxtable[err] = flux_err*convert
        #else:
        #    fluxtable[err] = flux_err
        if "W" in mag and "WISE" not in mag and 'WFC3' not in mag:
            fluxtable.rename_column(mag,mag.replace("W","WISE"))
            fluxtable.rename_column(err,err.replace("W","WISE"))

    #For all other photometry:
    other_mags = np.setdiff1d(mag_cols, wisecols+vistacols)
    other_errs = np.setdiff1d(mag_errcols, wise_errcols+vista_errcols)

    for mag, err in zip(other_mags, other_errs):
        flux, flux_err = _mags_to_flux(photometry_table[mag], mag_err = photometry_table[err])
        badflux = flux == -99.
        fluxtable[mag][badflux] = flux[badflux]
        fluxtable[mag][~badflux] = flux[~badflux]*convert
        #if flux != -99.:
        #    fluxtable[mag] = flux*convert
        #else:
        #    fluxtable[mag] = flux
        baderr = flux_err == -99.0
        fluxtable[err][baderr] = flux_err[baderr]
        fluxtable[err][~baderr] = flux_err[~baderr]*convert

        # Upper limits -- Assume to have been recorded as 3 sigma
        #   Arbitrarily set the value to 1/3 of the error (could even set to 0)
        uplimit = photometry_table[err] == 999.
        fluxtable[err][uplimit] = fluxtable[mag][uplimit] / 3.
        fluxtable[mag][uplimit] = fluxtable[mag][uplimit] / 9.
    return fluxtable
    
def specdb_to_marz(dbfile:str, specsource:str, marzfile:str=None)->fits.HDUList:
    """
    Take in a specdb file and convert it so that it
    can be read by MARZ.
    Args:
        dbfile (str): Path to a specDB hdf5 file.
        specsource (str): Source of spectrum. Should be one of
            groups in the specdb file. E.g. DEIMOS, GMOS-S, MUSE etc. All
            spectra are read if a source is not specified.
        marzfile (str, optional): Path to the marz fits file to be written.
            If it is not specified, then the fits file will be named
            <specsource>_marz.fits and will be written to the current working directory.
        
    Returns:
        marz_hdu (fits.HDUList): The fits hdulist that will be written.
    """
    # Validation
    assert os.path.isfile(dbfile), "File not found: {:s}".format(dbfile)
    
    # Read
    hdf = h5py.File(dbfile)

    # More validation and prepare for parsing spectra
    assert specsource in hdf.keys(), "{:s} is not present in the given dbfile".format(specsource)
    print("Reading {:s} spectra ...".format(specsource))
    # TODO: Make it so that all sources can be read and written to a MARZ file
    # At the moment, because the spectra in different sources have diffferent
    # array sizes, this is not implemented. Need to figure out how to pad correctly.

    # Parsing
    spectab = hdf[specsource]['spec']
    metatab = hdf[specsource]['meta']
    wave = spectab['wave']
    flux = spectab['flux']
    sig = spectab['sig']
    # Because sky spectrum is not stored in specDB at the moment,
    sky = np.zeros_like(wave)

    if marzfile is None:
        marzfile = specsource+"_marz.fits"


    # Instantiate fits hdus
    extnames = ['INTENSITY', 'VARIANCE', 'SKY', 'WAVELENGTH']
    datalist = [flux, sig**2, sky, wave]
    marz_hdu = fits.HDUList()

    for ext, data in zip(extnames, datalist):
        hdu = fits.ImageHDU(data)
        hdu.header.set('extname', ext)
        marz_hdu.append(hdu)
    # Add the meta table at the end
    marz_hdu.append(fits.BinTableHDU(np.array(metatab)))
    marz_hdu.writeto(marzfile, overwrite=True)

    return marz_hdu

    '''
    TODO: Write this function once CDS starts working again (through astroquery) 
    def xmatch_gaia(catalog,max_sep = 5*u.arcsec,racol='ra',deccol='dec'):
        """
        Cross match against Gaia DR2
        and return the cross matched table.
        Args:
            max_sep (Angle): maximum separation to be
                            considered a valid match.
        Returns:
            xmatch_tab (Table): a table with corss matched
                                entries.
        """
    ''' 
