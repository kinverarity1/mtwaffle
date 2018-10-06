'''General MT functions.'''
import logging
import os.path
import textwrap

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import numpy.linalg as LA
import scipy
from scipy.interpolate import interp1d
import scipy.optimize
import attrdict


logger = logging.getLogger(__name__)
RAD2DEG = 180 / np.pi


def mrad2deg(arr):
    '''Convert milliradians to degrees, and keep it in the first quadrant.'''
    arr = arr / 1000 / np.pi * 180
    arr2 = np.empty_like(arr)
    for i, d in enumerate(arr):
        while d < -90:
            d += 180
        arr2[i] = d
    return arr2


def linear_interp(freqs, Zs, newfreqs, extrapolation='remove'):
    '''Calculate impedance tensors at new frequencies by linear interpolation.

    Args:
        - *freqs*: n x 1 ndarray of frequencies
        - *Zs*: n x 2 x 2 complex ndarray of impedance tensors
        - *newfreqs*: n x 1 ndarray of new frequencies
        - *extrapolation*: string, one of:
            - 'nan': add np.NaN values for frequencies that require extrapolation;
              this guarantees newsfreqs.shape==freqs.shape and newZs.shape==Zs.shape
            - 'remove': alter *newfreqs* such that no extrapolation is done
            - 'error': raise Exception in extrapolation is required

    Returns:
        - *newfreqs*: m x 1 ndarray of new frequencies
        - *newZs*: m x 2 x 2 complex ndarray of impedance tensors

    '''
    freqs = np.asarray(freqs)
    newfreqs = np.asarray(newfreqs)
    assert len(freqs) == Zs.shape[0]

    # Sort Zs from low to high freq.
    indices = np.argsort(freqs)
    freqs = freqs[indices]
    Zs = Zs[indices]

    freq0 = freqs[0]
    freq1 = freqs[-1]

    if extrapolation == 'nan':
        Znan = np.ones((2, 2)) * np.nan
        for f in newfreqs:
            if f < freq0:
                freqs = np.insert(freqs, 0, f, axis=0)
                Zs = np.insert(Zs, 0, Znan, axis=0)
            if f > freq1:
                freqs = np.append(freqs, [f], axis=0)
                Zs = np.append(Zs, np.array([Znan]), axis=0)
        indices = np.argsort(freqs)
        freqs = freqs[indices]
        Zs = Zs[indices]
    elif extrapolation == 'remove':
        newfreqs = np.array([
                f for f in newfreqs if f >= freqs[0] and f <= freqs[-1]])
        newfreqs.sort()
    elif extrapolation == 'error':
        for nf in newfreqs:
            if nf < freqs[0]:
                raise Exception('newfreq %f < (%f-%f)' % (nf, freqs[0], freqs[-1]))
            if nf > freqs[-1]:
                raise Exception('newfreq %f > (%f-%f)' % (nf, freqs[0], freqs[-1]))

    newZs = np.empty((len(newfreqs), 2, 2), dtype=np.complex)
    for i, j in ((0,0), (0,1), (1,0), (1,1)):
        newZs[:,i,j] = (interp1d(freqs, Zs[:,i,j].real, axis=0)(newfreqs) +
                        interp1d(freqs, Zs[:,i,j].imag, axis=0)(newfreqs) * 1j)

    return newfreqs, newZs


def between_freqs(freqs, f0=None, f1=None):
    '''Return indices to freqs for items which are between two extremes (f0 and f1).

    Args:
        - *freqs*: n x 1 ndarray
        - *f0, f1*: floats for min and max frequencies

    Returns: *indices* to *freqs* array

    '''
    freqs = np.asarray(freqs)
    if f1 is None or f1 > np.max(freqs):
        f1 = np.max(freqs)
    if f0 is None or f0 < np.min(freqs):
        f0 = np.min(freqs)
    indices = []
    for i, freq in enumerate(freqs):
        if freq >= f0 and freq <= f1:
            indices.append(i)
    return np.asarray(indices)


def ohms2mV_km_nT(zs):
    '''Convert imp. tensor(s) from ohms to mV/km/nT.'''
    return zs * 796.


def mV_km_nT2ohms(zs):
    '''Convert imp. tensor(s) from mV/km/nT to ohms'''
    return zs / 796.


def inv_imag_sign(zs):
    '''Invert sign of imaginary parts of imp. tensor(s).'''
    return zs.real + zs.imag * -1 * 1j


def delete(arrays, indices):
    '''Delete *indices* from each ndarray in *arrays*.

    See source and ``np.delete`` function.

    '''
    ret_arrays = []
    for array in arrays:
        ret_arrays.append(np.delete(array, indices, axis=0))
    return ret_arrays


def delete_freq(del_freqs, freqs, arrays, ret_indices=False):
    '''Find the indices of *del_freqs* in *freqs* and delete those entries from
    each array in *arrays*, and return the new set of frequencies and arrays.

    Args:
        - *del_freqs*: frequencies to delete from *freqs*
        - *freqs*: sequence of frequencies
        - *arrays*: sequence of ndarrays

    Returns:
        - *freqs*: an ndarray of frequencies
        - *new_arrays*: a list of the passed *arrays* with the right thing removed.
        - (optional) *indices*: indices which were removed.

    '''
    new_freqs = list(freqs)
    for del_freq in del_freqs:
        if del_freq in freqs:
            i = new_freqs.index(utils.find_nearest(del_freq, freqs))
            del new_freqs[i]
            arrays = delete(arrays, i)
    if ret_indices:
        fdels = utils.find_nearest(del_freqs, freqs)
        indices = [list(freqs).index(fdel) for fdel in fdels]
        return np.array(new_freqs), arrays, indices
    else:
        return np.array(new_freqs), arrays


def appres(zs, freqs):
    '''Convert imp. tensor(s) (mV/km/nT) to apparent resistivity(s) (ohm.m).

    Args:
        - *freqs*: float or n x 1 ndarray
        - *zs*: float, 2 x 2 complex ndarray or n x 2 x 2 complex ndarray with
          impedance in units of mV/km/nT

    Returns: *res*
        - *res*: same shape as *zs*

    '''
    Zs = np.asarray(zs)
    try:
        assert Zs.ndim == 3
        res = np.empty_like(Zs, dtype=np.float)
        assert len(freqs) == Zs.shape[0]
        for i, f in enumerate(freqs):
            res[i, ...] = 0.2 / f * np.abs(Zs[i]) ** 2
        return res
    except:
        return 0.2 / freqs * np.abs(Zs) ** 2


def phase(zs):
    '''Phase of imp. tensor(s) - calculated in the first quadrant.'''
    return np.arctan(zs.imag / zs.real) * RAD2DEG


def phase2(zs):
    '''Phase of imp. tensor(s) - calculated with quadrant information preserved.'''
    return np.arctan2(zs.imag, zs.real) * RAD2DEG


def phase_abs(zs):
    '''Phase of imp. tensor(s) - forced into the first quadrant.'''
    return np.arctan(np.abs(zs.imag / zs.real)) * RAD2DEG


def rot(A, theta=0):
    '''Rotate 2 x 2 array A by *theta* degrees.'''
    t = np.float(theta) / RAD2DEG
    R = np.array([[np.cos(t), -1 * np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.dot(R.T, np.dot(A, R))


def rot_arr(arrs, theta):
    '''Rotate a list of 2 x 2 arrays by theta degrees.

    Arguments:
        arrs (list): list of 2 x 2 arrays.
        theta (int): degrees.

    '''
    return np.array([rot(arr, theta) for arr in arrs])


def lilley_Z1(z):
    return (z[0, 0] + z[1, 1]) / 2


def lilley_Z2(z):
    return (z[0, 0] - z[1, 1]) / 2


def lilley_Z3(z):
    return (z[0, 1] + z[1, 0]) / 2


def lilley_Z4(z):
    return (z[0, 1] - z[1, 0]) / 2


def Z3(z):
    return (z[0, 1] + z[1, 0]) / 2


def Z4(z):
    return (z[0, 0] - z[1, 1]) / 2


def tan4t(z, bit='both'):
    Z4cc = Z4(z).real + Z4(z).imag * -1j
    num = 2 * (Z3(z) * Z4cc).real
    den = np.abs(Z4(z)) ** 2 - np.abs(Z3(z)) ** 2
    if bit == 'both':
        return num / den
    elif bit == 'num':
        return num
    elif bit == 'den':
        return den


def egt(z):
    num = tan4t(z, 'num')
    den = tan4t(z, 'den')
    return np.arctan2(num, den) / 4


def fm9(z):
    return np.abs(z[0,1]) ** 2 + np.abs(z[1,0]) ** 2


def ptensors(zs):
    '''Calculate phase tensor(s) for imp. tensor(s) (Caldwell 2004).

    Arguments:
        zs (either 2 x 2 ndarray or [<2x2 ndarray>, <2x2 ndarray>, ...]): impedance tensors

    Returns: phase tensors in the same shape as the argument Z.

    '''
    Z = np.asarray(zs)
    if Z.ndim == 2:
        return np.dot(LA.inv(Z.real), Z.imag)
    elif Z.ndim == 3:
        return np.asarray([ptensors(Zi) for Zi in Z])


def ptens_normskew(zs):
    '''Normalised phase tensor skew(s) of Booker (2012).

    zs can be either 2 x 2 or n x 2 x 2 for n frequencies.

    '''
    Z = np.asarray(zs)
    if Z.ndim == 2:
        P = ptensors(Z)
        return np.arctan2(P[0, 1] - P[1, 0], np.trace(P)) * RAD2DEG
    elif Z.ndim == 3:
        return np.asarray([ptens_normskew(Zi) for Zi in Z])


def ptens_azimuth(zs):
    '''Rotation azimuth of phase tensor(s) such that diags are max-ed & Pxx > Pyy.

    Find the rotation angle for impedance tensor *Z* such that

     1. The sum of squares of the off-diagonals of the phase tensor are minimized
        (i.e. coordinate axes parallel to ellipse axes); and
     2. ptens[0, 0] > ptens[1, 1]
        (i.e. ellipse major axis is parallel to the first coordinate axis)

    (mathematical rotation angle, so it's counter-clockwise,
    but then the coordinate system is the reverse.)

    '''
    Z = np.asarray(zs)
    if Z.ndim == 2:
        def offdiagsum(t):
            x = rot(Z, t)
            P = ptensors(x)
            return P[0, 1] ** 2 + P[1, 0] ** 2

        xopt = scipy.optimize.fmin(offdiagsum, 0.1, disp=False)
        angle1 = xopt[0]
        logger.debug('ptensaz: inital solution=%f' % angle1)

        # We want the angle which aligns the 1st coordinate axis with the major
        # axis of the ellipse, so need to check the angle 90 degrees away from the
        # solution.

        if angle1 < 0:
            angle1 = 360 + angle1
        logger.debug('ptensaz: %f' % angle1)
        angle2 = angle1 - 90
        if angle2 < 0:
            angle2 = 360 + angle2
        logger.debug('ptensaz: after removal of negative angles=%f, %f' % (angle1, angle2))

        # We want the smaller angle, between 0 and 180 degrees:

        if angle1 > 180:
            angle1 -= 180
        if angle2 > 180:
            angle2 -= 180
        logger.debug('ptensaz: after adjustment to first 2 quadrants=%f, %f' % (angle1, angle2))

        ptens1 = ptensors(rot(Z, angle1))
        ptens2 = ptensors(rot(Z, angle2))
        if ptens2[0, 0] > ptens1[0, 0]:
            return angle2
        else:
            return angle1
    elif Z.ndim == 3:
        return np.array([ptens_azimuth(zi) for zi in Z])


def ptens_alpha(ptensors):
    '''Phase tensor(s) alpha angle (Caldwell 2004).'''
    P = np.asarray(ptensors)
    if P.ndim == 2:
        return 0.5 * np.arctan2((P[0,1] + P[1,0]), (P[0,0] - P[1,1])) * 180 / np.pi
    elif P.ndim == 3:
        return np.array([ptens_alpha(pi) for pi in P])


def ptens_beta(ptensors):
    '''Phase tensor(s) beta angle (Caldwell 2004).'''
    P = np.asarray(ptensors)
    if P.ndim == 2:
        return 0.5 * np.arctan2((P[0,1] - P[1,0]), (P[0,0] + P[1,1])) * 180 / np.pi
    elif P.ndim == 3:
        return np.array([ptens_beta(pi) for pi in P])


def ptens_min(ptensors):
    '''Minimum angle of phase tensor(s) (Caldwell 2004, A8).'''
    P = np.asarray(ptensors)
    if P.ndim == 2:
        return (np.sqrt(ptens1(P)**2 + ptens3(P)**2)
                - np.sqrt(ptens1(P)**2 + ptens3(P)**2 - ptens2(P)**2))
    elif P.ndim == 3:
        return np.array([ptens_min(pi) for pi in P])


def ptens_max(ptensors):
    '''Maximum angle of phase tensor(s) (Caldwell 2004, A9).'''
    P = np.asarray(ptensors)
    if P.ndim == 2:
        return (np.sqrt(ptens1(P)**2 + ptens3(P)**2)
                + np.sqrt(ptens1(P)**2 + ptens3(P)**2 - ptens2(P)**2))
    elif P.ndim == 3:
        return np.array([ptens_max(pi) for pi in P])
    

def ptens1(P):
    return ptens_tr(P) / 2.


def ptens2(P):
    return np.sqrt(ptens_det(P))


def ptens3(P):
    return ptens_skew(P) / 2.


def ptens_tr(P):
    return P[0, 0] + P[1, 1]


def ptens_skew(ptensors):
    '''Skew angle of phase tensor(s) (Caldwell 2004).'''
    P = np.asarray(ptensors)
    if P.ndim == 2:
        return P[0, 1] - P[1, 0]
    elif P.ndim == 3:
        return np.array([ptens_skew(pi) for pi in ptensors])


def ptens_det(P):
    return (P[0, 0] * P[1, 1]) - (P[0, 1] * P[1, 0])


def ptens_theta(ptensors):
    '''Theta angle from phase tensor(s).'''
    return ptens_alpha(P) - ptens_beta(P)


def ptens_ppspl(P):
    '''Return difference in degrees between Pmax and Pmin.'''
    p1 = np.rad2deg(np.arctan(ptens_max(P)))
    p0 = np.rad2deg(np.arctan(ptens_min(P)))
    return p1 - p0


def ptens_vectors(ptensors, n_thetas=45):
    '''Return n_theta vectors for phase tensor/s around the unit circle.

    For each vector v_u on the unit circle (there are n_thetas of these vectors)
    calculate P dot v_u and return the family of the resulting vectors, together
    with the thetas

    Returns: tuple (thetas, vecs)
        thetas (ndarray): the angles on the unit circle. Shape is (n_thetas).
        vecs (ndarray): the vectors. If ptensors.shape == (2, 2) then vecs.shape
            == (n_thetas, 2); if ptensors.shape == (m, 2, 2) then vecs.shape ==
            (m, n_thetas, 2).

    '''
    P = np.asarray(ptensors)
    if P.ndim == 2:
        thetas = np.linspace(0, 2 * np.pi, n_thetas)
        vecs = np.empty((n_thetas, 2))
        for i, t in enumerate(thetas):
            vunit = np.array([np.cos(t), np.sin(t)])
            vecs[i, ...] = np.dot(P, vunit)
        return thetas, vecs
    elif P.ndim == 3:
        vecs_list = []
        for pi in P:
            thetas, vecs = ptens_vectors(pi)
            vecs_list.append(vecs)
        return thetas, np.asarray(vecs_list)


# def ptens_misfit(thetas, obs_vecs, fwd_vecs):
#     '''Return phase tensor misfit vectors and angular misfits.

#     Args:
#         - *thetas*: n x 1 ndarray of angles
#         - *obs_vecs*: n x 2 ndarray from :func:`ptens_vectors`
#         - *fwd_vecs*: n x 2 ndarray from :func:`ptens_vectors`

#     Returns:
#         - *mf_vecs*: n x 2 ndarray of misfit vectors
#         - *mf_angles*: n x 1 ndarray of misfit angles between the observed and
#           forward resulting vector

#     '''
#     n = len(thetas)
#     mf_vecs = np.empty((n, 2))
#     mf_angles = np.empty(n)
#     for k, t in enumerate(thetas):
#         vd = obs_vecs[k]
#         vf = fwd_vecs[k]


def normfreqs(zs, freqs):
    '''Normalise imp. tensor(s) magnitude by multiplying by sqrt(period).'''
    Z = np.asarray(zs).copy()
    factor = np.sqrt(1. / freqs)
    if Z.ndim == 3:
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            Z[:, i, j] = Z[:, i, j].real * factor + Z[:, i, j].imag * factor * 1j
    else:
        Z = Z.real * factor + Z.imag * factor * 1j
    return Z


def bostick(freqs, appres, phases):
    '''Bostick transform of imp. tensor(s) - returns tuple (depths, resistivities).

    Args:
        - *freqs*: n x 1 ndarray
        - *res*: n x 2 x 2 or n x 1 ndarray of apparent resistivities
        - *phase*: ndarray, same shape as *res*, units of degrees

    Returns:
        - *bz*: n x m ndarray of depths in metres
        - *br*: n x m ndarray of resistivities in ohm metres

    '''
    freqs = np.asarray(freqs)
    appres = np.asarray(appres)
    phases = np.asarray(phases)
    n = len(freqs)
    if appres.shape == (n, 2, 2):
        bz = np.empty((n, 2, 2))
        for i in (0, 1):
            for j in (0, 1):
                bz[:, i, j] = 355.4 * np.sqrt(appres[:, i, j] / freqs)
    else:
        assert appres.shape == freqs.shape
        bz = 355.4 * np.sqrt(appres / freqs)
    br = appres * (3.1416 / (2 * np.deg2rad(phases)) - 1)
    return np.array([bz, br])


def z11b(z, b):
    return z[0, 0] * (np.cos(b) ** 2) + (z[0, 1] + z[1, 0]) * np.cos(b) * np.sin(b) + z[1, 1] * (np.sin(b) ** 2)


def z12b(z, b):
    return z[0, 1] * (np.cos(b) ** 2) + (z[1, 1] - z[0, 0]) * np.cos(b) * np.sin(b) - z[1, 0] * (np.sin(b) ** 2)


def cgamma(Z, out_unit='deg'):
    '''Invariant measure of 3D-ness of impedance tensor (Lilley 1998, eq. 37).'''
    return catan2(Z[1, 1] + Z[0, 0], Z[0, 1] - Z[1, 0], out_unit)


def cgammas(zs, out_unit='deg'):
    '''Invariant measure of 3D-ness of impedance tensors (Lilley 1998, eq. 37).'''
    return np.array([cgamma(Z, out_unit) for Z in Zs])


def pos_quads(carr, units='deg'):
    '''Move angles from the 3rd and 4th quadrants into the 1st or 2nd quadrants,
    using the opposite direction.'''
    if units == 'deg':
        opp = 180
    else:
        opp = np.pi
    carr_re = carr.real
    carr_im = carr.imag
    for i in range(len(carr)):
        if carr_re[i] < 0:
            carr_re[i] += opp
        if carr_im[i] < 0:
            carr_im[i] += opp
    return carr_re + carr_im * 1j


def catan2(num, den, out_unit='deg'):
    '''Complex arctan2 function.

    Arguments:
        num (float): numerator
        den (float): denominator
        out_unit (str): either 'deg' or 'rad'

    '''
    real = np.arctan2(num.real, den.real)
    imag = np.arctan2(num.imag, den.imag)
    if out_unit == 'deg':
        real = real * 180 / np.pi
        imag = imag * 180 / np.pi
    else:
        assert out_unit == 'rad'
    return real + imag * 1j


lzdd = lambda z: z[1, 1] - z[0, 0]
lzos = lambda z: z[0, 1] + z[1, 0]
lzds = lambda z: z[1, 1] + z[0, 0]
lzod = lambda z: z[0, 1] - z[1, 0]


def theta_e(z, out_unit='deg'):
    '''Electric strike of impedance tensor (Lilley 1998).

    Arguments:
        out_unit (str): 'deg' or 'rad'

    '''
    return 0.5 * (catan2(lzdd(z), lzos(z), out_unit) + catan2(lzds(z), lzod(z), out_unit))


def theta_h(z, out_unit='deg'):
    '''Magnetic strike of impedance tensor (Lilley 1998).

    Arguments:
        out_unit (str): 'deg' or 'rad'
    
    '''
    return 0.5 * (catan2(lzdd(z), lzos(z), out_unit) - catan2(lzds(z), lzod(z), out_unit))


def theta_es(zs, **kwargs):
    '''Electric strike of impedance tensors (Lilley 1998).

    See theta_e function for keyword arguments.

    '''
    return np.array([theta_e(z) for z in zs])


def theta_hs(zs, **kwargs):
    '''Magnetic strike of impedance tensors (Lilley 1998).

    See theta_h function for keyword arguments.

    '''
    return np.array([theta_h(z) for z in zs])


class L(object):
    def __init__(s, T):
        T11 = T[0, 0]
        T12 = T[0, 1]
        T21 = T[1, 0]
        T22 = T[1, 1]
        s.t1 = (T11 + T22) / 2
        s.t2 = (T12 + T21) / 2
        s.t3 = (T11 - T22) / 2
        s.t4 = (T12 - T21) / 2
        s.t0 = np.sqrt(s.t2 ** 2 + s.t3 ** 2)


def t11b(z, b):
    return z[0, 0] * (np.cos(b) ** 2) + (z[0, 1] + z[1, 0]) * np.cos(b) * np.sin(b) + z[1, 1] * (np.sin(b) ** 2)


def t12b(z, b):
    return z[0, 1] * (np.cos(b) ** 2) + (z[1, 1] - z[0, 0]) * np.cos(b) * np.sin(b) - z[1, 0] * (np.sin(b) ** 2)




callables = {fname: globals()[fname] for fname in dir() if callable(globals()[fname])}
