""" Methods related to fussing with a catalog"""
import numpy as np
import pdb

from astropy.coordinates import SkyCoord
from astropy import units

# Import check
try:
    from astroquery.heasarc import Heasarc
except ImportError:
    print("Warning: You need to install astroquery to use the survey tools...")
else:
    # Instantiate
    heasarc = Heasarc()


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


