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


logger = logging.getLogger(__name__)

RAD2DEG = 180 / np.pi

from mtwaffle.utils import AttrDict


def show_indices(arr):
    r"""Return a string showing positive and negative indices for elements of a
    list.

    For example::

        >>> print(arr)
        [  1.0883984    1.52735318   2.14333993   3.00775623   4.22079456
           5.92305539   8.31184382  11.66403876  16.36818533]
        >>> print("\n".join(mt.show_indices(arr)))
        0 [-9] = 1.088
        1 [-8] = 1.527
        2 [-7] = 2.143
        3 [-6] = 3.008
        4 [-5] = 4.221
        5 [-4] = 5.923
        6 [-3] = 8.312
        7 [-2] = 11.664
        8 [-1] = 16.368

    """
    return ["%d [-%d] = %5.3f" % (i[0], len(arr) - i[0], i[1])
            for i in zip(range(len(arr)), arr)]


def linear_interp(freqs, Zs, newfreqs, extrapolation='remove'):
    """Calculate impedance tensors at new frequencies by linear interpolation.

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

    """
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
    """Return impedance tensors between frequencies, inclusive.

    Args:
        - *freqs*: n x 1 ndarray
        - *f0, f1*: floats for min and max frequencies

    Returns: *indices* to *freqs* array
    """
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

def ohms2mV_km_nT(Z):
    """Convert ohms to mV/km/nT."""
    return Z * 796.


def mV_km_nT2ohms(Z):
    """Convert mV/km/nT to ohms"""
    return Z / 796.


def inv_imag_sign(Z):
    """Invert sign of imaginary parts of Z."""
    return Z.real + Z.imag * -1 * 1j


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


def appres(freqs, Zs):
    """Calculate apparent resistivity.

    Args:
        - *freqs*: float or n x 1 ndarray
        - *Zs*: float, 2 x 2 complex ndarray or n x 2 x 2 complex ndarray with
          impedance in units of mV/km/nT

    Returns: *res*
        - *res*: same shape as *Zs*
    """
    Zs = np.asarray(Zs)
    try:
        assert Zs.ndim == 3
        res = np.empty_like(Zs, dtype=np.float)
        assert len(freqs) == Zs.shape[0]
        for i, f in enumerate(freqs):
            res[i, ...] = 0.2 / f * np.abs(Zs[i]) ** 2
        return res
    except:
        return 0.2 / freqs * np.abs(Zs) ** 2


def phase(Zs):
    """Phase in the first quadrant."""
    return np.arctan(Zs.imag / Zs.real) * RAD2DEG


def phase2(Zs):
    """Phase with quadrant information preserved."""
    return np.arctan2(Zs.imag, Zs.real) * RAD2DEG


def phase_abs(Zs):
    """Phase forced to be in the first quadrant."""
    return np.arctan(np.abs(Zs.imag / Zs.real)) * RAD2DEG


def rot(A, theta=0):
    """Rotate 2 x 2 array A by *theta* degrees."""
    t = np.float(theta) / RAD2DEG
    R = np.array([[np.cos(t), -1 * np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.dot(R.T, np.dot(A, R))


def rot_arr(arrs, theta):
    return np.array([rot(arr, theta) for arr in arrs])

def ptens(Z):
    """Phase tensor for either one or multiple impedance tensors."""
    Z = np.asarray(Z)
    if Z.ndim == 2:
        return np.dot(LA.inv(Z.real), Z.imag)
    elif Z.ndim == 3:
        return np.asarray([ptens(Zi) for Zi in Z])


def normptskew(Z):
    """Normalised phase tensor skew of Booker (2012).
    Z can be either 2 x 2 or n x 2 x 2 for n frequencies."""
    Z = np.asarray(Z)
    if Z.ndim == 2:
        P = ptens(Z)
        return np.arctan2(P[0, 1] - P[1, 0], np.trace(P)) * RAD2DEG
    elif Z.ndim == 3:
        return np.asarray([normptskew(Zi) for Zi in Z])

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

def ptensaz(Z):
    """Find the rotation angle for impedance tensor *Z* such that

    mathematical rotation angle, so it's counter-clockwise, but then the coordinate system is the reverse.

     1. The sum of squares of the off-diagonals of the phase tensor are minimized
        (i.e. coordinate axes parallel to ellipse axes); and
     2. ptens[0, 0] > ptens[1, 1]
        (i.e. ellipse major axis is parallel to the first coordinate axis)

    """
    def offdiagsum(t):
        x = rot(Z, t)
        P = ptens(x)
        return P[0, 1] ** 2 + P[1, 0] ** 2

    xopt = scipy.optimize.fmin(offdiagsum, 0.1, disp=False)
    angle1 = xopt[0]
    logger.debug("ptensaz: inital solution=%f" % angle1)

    # We want the angle which aligns the 1st coordinate axis with the major
    # axis of the ellipse, so need to check the angle 90 degrees away from the
    # solution.

    if angle1 < 0:
        angle1 = 360 + angle1
    logger.debug("ptensaz: %f" % angle1)
    angle2 = angle1 - 90
    if angle2 < 0:
        angle2 = 360 + angle2
    logger.debug("ptensaz: after removal of negative angles=%f, %f" % (angle1, angle2))

    # We want the smaller angle, between 0 and 180 degrees:

    if angle1 > 180:
        angle1 -= 180
    if angle2 > 180:
        angle2 -= 180
    logger.debug("ptensaz: after adjustment to first 2 quadrants=%f, %f" % (angle1, angle2))

    ptens1 = ptens(rot(Z, angle1))
    ptens2 = ptens(rot(Z, angle2))
    if ptens2[0, 0] > ptens1[0, 0]:
        return angle2
    else:
        return angle1


def ptensazimuths(Zs):
    """Return phase tensor azimuths for several impedance tensors. See
    :func:`ptensaz`."""
    return np.array([ptensaz(Z) for Z in Zs])


def sites_data(sites, phase_func=phase):
    '''Return components of array.

    Args:
        - *sites*: list containing dicts with keys freqs, zs, zes

    '''
    data = []
    for i, site in enumerate(sites):
        data.append(site_data(site, phase_func=phase_func))
    return data

def calc_basic_props(freqs, zs, phase_func=phase, fillAttrDict=None):
    '''Calculate the basic properties of the MT impedance.

    Arguments:
        freqs (ndarray shape=(n,) dtype=float): frequencies
        zs (ndarray shape=(n, 2, 2) dtype=complex): complex impedance tensors
            in mV/km/nT

    Returns: an AttrDict with 'res_xy', 'phase_yx', 'zr_xx', 'zi_yy'
    keys, etc.

    '''
    if fillAttrDict is None:
        sd = AttrDict()
    else:
        sd = fillAttrDict
    sd["freqs"] = np.asarray(freqs)
    sd["zs"] = np.asarray(zs)
    res = appres(sd['freqs'], sd['zs'])
    phase = phase_func(sd['zs'])
    for key, indices in zip(('xx', 'xy', 'yx', 'yy'), ([0, 0], [0, 1], [1, 0], [1, 1])):
        sd['res_' + key] = res[[Ellipsis] + indices]
        sd['phase_' + key] = phase[[Ellipsis] + indices]
        sd['zr_' + key] = sd['zs'].real[[Ellipsis] + indices]
        sd['zi_' + key] = sd['zs'].imag[[Ellipsis] + indices]
        sd['ptens'] = ptens(sd['zs'])
        sd['ptensaz'] = ptensazimuths(sd['zs'])
        sd['normptskew'] = normptskew(sd['zs'])
    return sd


def sd_asarr(sitesd, key='res_xy', imshowtr=False):
    arr = np.empty((len(sitesd), len(sitesd[0]['freqs'])))
    for si, site in enumerate(sitesd):
        for fi, freq in enumerate(site['freqs']):
            arr[si, fi] = site[key][fi]
    if imshowtr:
        arr = np.flipud(arr.T)
    return arr


def plot_ptensell(ptensors, freqs=None, scale=1, x0=0, y0=0, centre_dot=False,
                  xscale=1, color="k", lw=1, fmt="%s", resolution=20, xlabstep=1,
                  fig=None, fign=None, ax=None, colours=None, rot90=False):
    """Plot phase tensor ellipses.

        - *ptensors*: n x 2 x 2 ndarray of n phase tensors
        - *freqs*: optional n x 1 ndarray of frequencies
        - *scale*: size of the "unit" circle radius
        - *x0, y0*: location of phase tensor ellipse centre.
        - *centre_dot*: bool
        - *xscale*: control horizontal (frequency axis) spacing of ellipses
        - *resolution*: number of angles to use in drawing the ellipses


    """
    if ax is None:
        if fig is None:
            fig = plt.figure(fign)#, figsize=(15, 4))
        ax = fig.add_subplot(111, aspect="equal")
    if not freqs is None:
        sortis = np.asarray(np.argsort(freqs)[::-1], dtype=int)
        freqs = np.asarray(freqs)[sortis]
        ptensors = ptensors[sortis]
    if not colours:
        colours = [color] * len(ptensors)
    for pi in range(len(ptensors)):
        P = ptensors[pi]
        thetas = np.linspace(0, 1.7 * np.pi, resolution)
        x = np.zeros(len(thetas))
        y = np.zeros(len(thetas))
        #P = np.rot90(P)
        for i, theta in enumerate(thetas):
            x1 = scale * np.cos(theta)
            y1 = scale * np.sin(theta)
            x2, y2 = np.dot(P, (x1, y1))
            x[i] = x2 + x0 + pi * xscale
            y[i] = y2 + y0
        c = colours[pi]
        ax.plot(x[:1], y[:1], marker='.', mfc='k', )
        ax.plot(x, y, ls="-", color=c, lw=lw)
        if centre_dot:
            ax.plot(x0 + pi * xscale, y0, marker=",", color=color)
    if not freqs is None:
        indices = range(0, len(ptensors), xlabstep)
        ax.set_xticks(indices)
        ax.set_xticklabels(map(lambda f: fmt % f, freqs[indices]))
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")


def plot_ptensell_filled(ptensors, freqs=None, x0=0, y0=0,
                         fillarr=None, cmap=plt.cm.spectral_r,
                         vmin=None, vmax=None,
                         facecolor='none', edgecolor='k',
                         fmt="%s", xlabstep=1,
                         adj_pmax=lambda x:x, adj_pmin=lambda x:x,
                         extra_rotation=0,
                         fig=None, fign=None, ax=None, plotkws=None):
    '''Plot phase tensor ellipses with filled centres.

    They are rotated by 90 deg...
    '''
    if plotkws is None:
        plotkws = {}
    ptensors = np.asarray(ptensors)
    if ax is None:
        if fig is None:
            fig = plt.figure(fign)#, figsize=(15, 4))
        ax = fig.add_subplot(111, aspect="equal")
    if not freqs is None:
        sortis = np.asarray(np.argsort(freqs)[::-1], dtype=int)
        freqs = np.asarray(freqs)[sortis]
        ptensors = ptensors[sortis]
    if not fillarr is None:
        fillarr = np.asarray(fillarr)
        assert fillarr.shape[0] == ptensors.shape[0]
        if vmin is None:
            vmin = np.nanmin(fillarr)
        if vmax is None:
            vmax = np.nanmax(fillarr)
        fillarr_norm = [(f-vmin)/(vmax-vmin) for f in fillarr]
        colours = [cmap(f) for f in fillarr_norm]
    else:
        colours = [facecolor] * ptensors.shape[0]
    for pi in range(len(ptensors)):
        P = ptensors[pi]
        pmax = adj_pmax(ptens_max(P))
        pmin = adj_pmin(ptens_min(P))
        alpha = ptens_alpha(P)
        beta = ptens_beta(P)

        if edgecolor == 'auto':
            ec = colours[pi]
        else:
            ec = edgecolor
        e = Ellipse((x0, y0), pmax*2, pmin*2, alpha-beta+extra_rotation,
                    edgecolor=ec, facecolor=colours[pi], **plotkws)
        ax.add_artist(e)
    if not freqs is None:
        indices = range(0, len(ptensors), xlabstep)
        ax.set_xticks(indices)
        ax.set_xticklabels(map(lambda f: fmt % f, freqs[indices]))
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")


def animate_ptens(P, fign=1, clf=True, pngs_path=None, axes="math"):
    """Animate the phase tensor P.

    Plot the product of P with a unit vector as it rotates around the unit
    circle. The unit vector's tip is blue dotted and the product with P is red.

    Args:
        - *P*: 2 x 2 real ndarray phase tensor
        - *axes*: tuple of which direction

    """
    plt.ion()
    fig = plt.figure(fign)
    if clf:
        fig.clf()
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")

    vrp = np.dot(P, [1., 0.])
    vtp = [1., 0.]
    logger.debug("animate_ptens: vrp=%s" % vrp)
    logger.debug("animate_ptens: vtp=%s" % vtp)
    for i, t in enumerate(np.linspace(0, 2 * np.pi, 45)):
        vt = np.array([np.cos(t), np.sin(t)])
        vr = np.dot(P, vt)
        plt.plot([vrp[1], vr[1]], [vrp[0], vr[0]], color="r", ls="-")
        plt.plot([vtp[1], vt[1]], [vtp[0], vt[0]], color="b", ls=":")
        plt.draw()
        if pngs_path:
            fn = os.path.join(pngs_path, "aniptens_%06.0f.png" % i)
            fig.savefig(fn, dpi=50)
        vrp = vr
        vtp = vt
    plt.ioff()


def animate_ptensors(Ps, colour="k", colours=None, lw=1, lws=None, fign=1,
                     clf=True, pngs_path=None, resolution=45, axes="normal",
                     unit_colour="g", unit_lw=1, unit_ls=":"):
    """Animate the phase tensor P.

    Plot the product of P with a unit vector as it rotates around the unit
    circle. The unit vector's tip is drawn with *unit_colour*, *unit_ls*, and
    *unit_lw*.

    Args:
        - *Ps*: n x 2 x 2 ndarray of n phase tensors
        - *colour*: matplotlib colour string
        - *lw*: matplotlib line width
        - *colours*: n x 1 list of colour strings, optional
        - *lws*: n x 1 list of floats for line widths
        - *pngs_path*: None or path to a folder in which to write PNG images
          for creating a animation or video
        - *resolution*: number of angles to use. This controls both the speed
          and the smoothness of the plot: 45 is high enough for a decent plot,
          but perhaps a little fast. Higher values will be smoother and slower.
        - *axes*: "normal" or "reversed" -- "normal" has the first axis of the
          phase tensor on the horizontal axis, and the second axis on the
          vertical, with positive angles in a counter-clockwise direction.
          "reversed" is the compass convention, with the first axis on the
          vertical axis, the second axis on the horizontal axis, and positive
          angles are clockwise.
        - *unit_colour, unit_ls, unit_lw*: control line style of unit circle
          vectors' tips.


    This function won't work with inline pylab mode in IPython. You can disable
    that by running ``%pylab`` beforehand (and re-enable it with
    ``%pylab inline``).

    """
    plt.ion()
    fig = plt.figure(fign)
    if clf:
        fig.clf()
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")

    Ps = np.asarray(Ps)
    n = Ps.shape[0]
    if colours is None:
        colours = "k" * n
    if lws is None:
        lws = [lw] * n
    vrps = np.empty((n, 2))
    for ni in range(n):
        vrps[ni, ...] = np.dot(Ps[ni], [1., 0.])
    vtp = [1., 0.]
    logger.debug("animate_ptens: vrps=%s" % vrps)
    logger.debug("animate_ptens: vtp=%s" % vtp)
    if axes == "normal":
        axes_indices = (0, 1)
    else:
        axes_indices = (1, 0)
    for i, t in enumerate(np.linspace(0, 2 * np.pi, resolution)):
        vt = np.array([np.cos(t), np.sin(t)])
        plt.plot([vtp[axes_indices[0]], vt[axes_indices[0]]],
                 [vtp[axes_indices[1]], vt[axes_indices[1]]],
                 color=unit_colour, ls=unit_ls, lw=unit_lw)
        vtp = vt
        for ni in range(n):
            vr = np.dot(Ps[ni], vt)
            plt.plot([vrps[ni, axes_indices[0]], vr[axes_indices[0]]],
                     [vrps[ni, axes_indices[1]], vr[axes_indices[1]]],
                     color=colours[ni], ls="-", lw=lws[ni])
            vrps[ni, ...] = vr
        plt.draw()
        if pngs_path:
            fn = os.path.join(pngs_path, "aniptens_%06.0f.png" % i)
            fig.savefig(fn, dpi=50)
    plt.ioff()


def ptens_alpha(P):
    return 0.5 * np.arctan2((P[0,1] + P[1,0]), (P[0,0] - P[1,1])) * 180 / np.pi

def ptens_beta(P):
    return 0.5 * np.arctan2((P[0,1] - P[1,0]), (P[0,0] + P[1,1])) * 180 / np.pi


def ptens_alphas(ptensors):
    return np.array([ptens_alpha(ptens) for ptens in ptensors])


def ptens_betas(ptensors):
    return np.array([ptens_beta(ptens) for ptens in ptensors])


def ptens_min(P):
    return (np.sqrt(ptens1(P)**2 + ptens3(P)**2)
            - np.sqrt(ptens1(P)**2 + ptens3(P)**2 - ptens2(P)**2))


def ptens_max(P):
    return (np.sqrt(ptens1(P)**2 + ptens3(P)**2)
            + np.sqrt(ptens1(P)**2 + ptens3(P)**2 - ptens2(P)**2))


def ptens1(P):
    return ptens_tr(P) / 2.


def ptens2(P):
    return np.sqrt(ptens_det(P))


def ptens3(P):
    return ptens_sk(P) / 2.


def ptens_tr(P):
    return P[0, 0] + P[1, 1]


def ptens_sk(P):
    return P[0, 1] - P[1, 0]


def ptens_det(P):
    return (P[0, 0] * P[1, 1]) - (P[0, 1] * P[1, 0])


def ptens_theta(P):
    return ptens_alpha(P) - ptens_beta(P)


def ptens_ppspl(P):
    '''Return difference in degrees between Pmax and Pmin.'''
    p1 = np.rad2deg(np.arctan(ptens_max(P)))
    p0 = np.rad2deg(np.arctan(ptens_min(P)))
    return p1 - p0


ptskew = np.frompyfunc(ptens_sk, 1, 1)
ptmax = np.frompyfunc(ptens_max, 1, 1)
ptmin = np.frompyfunc(ptens_min, 1, 1)
ptalpha = np.frompyfunc(ptens_alpha, 1, 1)
ptbeta = np.frompyfunc(ptens_beta, 1, 1)
pttheta = np.frompyfunc(ptens_theta, 1, 1)
ptppspl = np.frompyfunc(ptens_ppspl, 1, 1)

def ptens_vectors(P, n_thetas=45):
    """Return phase tensor vectors.

    For each vector v_u on the unit circle (there are n_thetas of these vectors)
    calculate P dot v_u and return the family of the resulting vectors, together
    with the thetas

    Returns:
        - *thetas* (on the unit circle)
        - *vecs*: n_thetas x 2 ndarray
    """
    thetas = np.linspace(0, 2 * np.pi, n_thetas)
    vecs = np.empty((n_thetas, 2))
    for i, t in enumerate(thetas):
        vunit = np.array([np.cos(t), np.sin(t)])
        vecs[i, ...] = np.dot(P, vunit)
    return thetas, vecs


def ptens_misfit(thetas, obs_vecs, fwd_vecs):
    """Return phase tensor misfit vectors and angular misfits.

    Args:
        - *thetas*: n x 1 ndarray of angles
        - *obs_vecs*: n x 2 ndarray from :func:`ptens_vectors`
        - *fwd_vecs*: n x 2 ndarray from :func:`ptens_vectors`

    Returns:
        - *mf_vecs*: n x 2 ndarray of misfit vectors
        - *mf_angles*: n x 1 ndarray of misfit angles between the observed and
          forward resulting vector

    """
    n = len(thetas)
    mf_vecs = np.empty((n, 2))
    mf_angles = np.empty(n)
    for k, t in enumerate(thetas):
        vd = obs_vecs[k]
        vf = fwd_vecs[k]


def normfreqs(Z, freqs):
    '''Normalise Z by multiplying by the square root of the period.'''
    Z = Z.copy()
    factor = np.sqrt(1 / freqs)
    if Z.ndim == 3:
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            Z[:, i, j] = Z[:, i, j].real * factor + Z[:, i, j].imag * factor * 1j
    else:
        Z = Z.real * factor + Z.imag * factor * 1j
    return Z


def plot_res_phase2(freqs, Zs, phase_func=phase, **kwargs):
    """Quick wrapper for plotting the two modes resistivity and phase.

    Args:
        - *freqs*: n x 1 ndarray
        - *Zs*: n x 2 x 2 complex ndarray
        - *phase_func*: function used to calculate phase

    Kwargs: passed to :func:`plot_res_phase`

    """
    res = appres(freqs, Zs)
    phase = phase_func(Zs)
    res_indiv_kws = kwargs.get("res_indiv_kws", [{}, {}])
    res_indiv_kws[0].update({"label": "xy", "color": "b"})
    res_indiv_kws[1].update({"label": "yx", "color": "g"})
    kwargs["res_indiv_kws"] = res_indiv_kws
    return plot_res_phase([freqs, freqs], [res[:, 0, 1], res[:, 1, 0]],
                          [phase[:, 0, 1], phase[:, 1, 0]], **kwargs)


def plot_res_phase(freqs, reses, phases, res_es=None, phase_es=None,
                   res_kws=None, phase_kws=None,
                   res_indiv_kws=None, phase_indiv_kws=None,
                   res0=None, res1=None, phase0=None, phase1=None,
                   f0=None, f1=None,
                   layout=None, fig=None, figsize=None, fign=None,
                   res_ax=None, phase_ax=None,
                   grid="both", legend="res"):
    """Plot resistivity and phase curves as a function of frequency.

    Args:
        - *freqs*: [freqs_1, freqs_2, ... freqs_n] list of m ndarrays
        - *reses*: [reses_1, reses_2, ... reses_n] list of m ndarrays
        - *phases*: [phases_1, phases_2, ... phases_n] list of m ndarrays
        - *res_es, phase_es*: lists of m error ndarrays in the same form as above
        - *res_kws, phase_kws*: base keyword argument dictionaries for calls
          to plt.errorbar() functions for resistivity and phase plots
          respectively
        - *res_indiv_kws, phase_indiv_kws*: [kws_1, kws_2, ... kws_n] list of m
          dictionaries of keyword arguments to update the base dictionaries
          for calls to plt.errorbar()
        - *res0, res1*: resistivity axis limits
        - *phase0, phase*: phase axis limits
        - *f0, f1*: frequency axis limits
        - *layout*: "vertical", "horizontal", None. Arrangement of resistivity
          and phase subplots.
        - *res_ax, phase_ax*: resistivity and phase Axes objects
        - *grid, legend*: "both", "res", "phase"

    Returns: *res_ax, phase_ax* matplotlib Axes objects

    """
    if layout:
        if layout.startswith("v"): # vertical
            sps = (211, 212)
            if figsize is None:
                figsize = (5, 10)
        elif layout.startswith("h"): # horizontal
            sps = (121, 122)
            if figsize is None:
                figsize = (10, 5)
    if res_ax is None or phase_ax is None:
        if fig is None:
            if figsize is None:
                figsize = (10, 5)
                sps = (121, 122)
            else:
                if figsize[1] > figsize[0]:
                    sps = (211, 212)
                else:
                    sps = (121, 122)
            fig = plt.figure(fign, figsize=figsize)
        res_ax = fig.add_subplot(sps[0])
        phase_ax = fig.add_subplot(sps[1])

    if res_kws is None:
        res_kws = {}
    if phase_kws is None:
        phase_kws = {}

    m = len(freqs)
    for i in range(m):
        fs = freqs[i]
        res = reses[i]
        phase = phases[i]
        if res_es:
            res_e = res_es[i]
        else:
            res_e = np.zeros(len(fs))
        if phase_es:
            phase_e = phase_es[i]
        else:
            phase_e = np.zeros(len(fs))

        if res_indiv_kws:
            res_kws_i = res_kws.copy()
            res_kws_i.update(res_indiv_kws[i])
        else:
            res_kws_i = res_kws
        if phase_indiv_kws:
            phase_kws_i = phase_kws.copy()
            phase_kws_i.update(phase_indiv_kws[i])
        else:
            phase_kws_i = phase_kws

        res_ax.errorbar(fs, res, **res_kws_i)
        phase_ax.errorbar(fs, phase, **phase_kws_i)

        del res_kws_i
        del phase_kws_i

    res_ax.set_xscale("log")
    res_ax.set_yscale("log")
    phase_ax.set_xscale("log")

    if res0 and res1:
        res_ax.set_ylim(res0, res1)
    if phase0 and phase1:
        phase_ax.set_ylim(phase0, phase1)

    freqs_flat = np.asarray(freqs).ravel()
    if f0 is None:
        f0 = np.min(np.ma.masked_invalid(freqs_flat))
    if f1 is None:
        f1 = np.max(np.ma.masked_invalid(freqs_flat))

    res_ax.set_xlim(f1, f0)
    phase_ax.set_xlim(f1, f0)

    if grid:
        res_ax.grid()
        phase_ax.grid()

    if legend == "res" or "both":
        res_ax.legend(loc="best")
    if legend == "phase" or "both":
        phase_ax.legend(loc="best")

    res_ax.set_ylabel(r"Apparent resistivity [$\Omega$m]")
    phase_ax.set_ylabel("Phase [deg]")

    res_ax.set_xlabel("Frequency [Hz]")
    phase_ax.set_xlabel("Frequency [Hz]")

    return res_ax, phase_ax


def plot_impedance_tensors(Zs, freqs=None, z0=None, z1=None, f0=None, f1=None,
                           real_kws=None, imag_kws=None, horiz_line_kws=None,
                           fig=None, fign=None, clf=True, normbyfreqs=False):
    """Plot impedance tensors.

    Args:
        - *Zs*: n x 2 x 2 complex ndarray of n impedance tensors
        - *freqs*: optional n x 1 ndarray of frequencies
        - *z0, z1*: optional limits for impedance y-axis
        - *f0, f1*: optional limits for frequency x-axis
        - *real_kws, imag_kws, horiz_line_kws*: matplotlib kwargs for plot calls

    """
    if not real_kws:
        real_kws = {}
    if not imag_kws:
        imag_kws = {}
    if not horiz_line_kws:
        horiz_line_kws = {}
    imag_kws.setdefault("ls", "--")
    horiz_line_kws.setdefault("ls", ":")
    horiz_line_kws.setdefault("color", "k")
    if not fig:
        fig = plt.figure(fign)
    if clf:
        fig.clf()
    # grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.25)
    grid = [fig.add_subplot(221),
            fig.add_subplot(222),
            fig.add_subplot(223),
            fig.add_subplot(224)]
    Zs = np.asarray(Zs)
    if normbyfreqs and freqs is None:
        normbyfreqs = False
    if z0 is None:
        z0 = np.nanmin((np.nanmin(Zs.real), np.nanmin(Zs.imag)))
    if z1 is None:
        z1 = np.nanmax((np.nanmax(Zs.real), np.nanmax(Zs.imag)))
    ylim = (z0, z1)
    if not freqs is None:
        freqs_prov = True
    else:
        freqs_prov = False
        freqs = range(Zs.shape[0])
    xlim = (np.max(freqs), np.min(freqs))
    label_dict = {0: "x", 1: "y"}
    max_z = -1e10
    min_z = 1e10
    for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        Zr = np.asarray(Zs[:, i, j].real)
        Zi = np.asarray(Zs[:, i, j].imag)
        if normbyfreqs:
            Zr *= np.sqrt(1 / freqs)
            Zi *= np.sqrt(1 / freqs)
        max_zij = np.nanmax([np.nanmax(Zr), np.nanmax(Zi)])
        min_zij = np.nanmin([np.nanmin(Zr), np.nanmin(Zi)])
        if max_zij > max_z:
            max_z = max_zij
        if min_zij < min_z:
            min_z = min_zij
        grid[k].plot(freqs, Zr, **real_kws)
        grid[k].plot(freqs, Zi, **imag_kws)
        if freqs_prov:
            grid[k].set_xscale("log")
        grid[k].set_xlim(*xlim)
        if ylim:
            grid[k].set_ylim(*ylim)
        title_txt = "%s%s" % (label_dict[i], label_dict[j])
        grid[k].set_title(title_txt)
        grid[k].axhline(0, **horiz_line_kws)
    if normbyfreqs and ylim is None:
        for k in range(4):
            grid[k].set_ylim(min_z, max_z)
    grid[0].set_xticks([])
    grid[1].set_xticks([])
    for ax in grid:
        ax.set_xlim(f1, f0)


def bostick(freqs, res, phase):
    '''Bostick transform.

    Args:
        - *freqs*: n x 1 ndarray
        - *res*: n x 2 x 2 or n x 1 ndarray of apparent resistivities
        - *phase*: ndarray, same shape as *res*, units of degrees

    Returns:
        - *bz*: n x m ndarray of depths in metres
        - *br*: n x m ndarray of resistivities in ohm metres

    '''
    freqs = np.asarray(freqs)
    res = np.asarray(res)
    phase = np.asarray(phase)
    n = len(freqs)
    if res.shape == (n, 2, 2):
        bz = np.empty((n, 2, 2))
        for i in (0, 1):
            for j in (0, 1):
                bz[:, i, j] = 355.4 * np.sqrt(res[:, i, j] / freqs)
    else:
        assert res.shape == freqs.shape
        bz = 355.4 * np.sqrt(res / freqs)
    br = res * (3.1416 / (2 * np.deg2rad(phase)) - 1)
    return bz, br

def z11b(z, b):
    return z[0, 0] * (np.cos(b) ** 2) + (z[0, 1] + z[1, 0]) * np.cos(b) * np.sin(b) + z[1, 1] * (np.sin(b) ** 2)

def z12b(z, b):
    return z[0, 1] * (np.cos(b) ** 2) + (z[1, 1] - z[0, 0]) * np.cos(b) * np.sin(b) - z[1, 0] * (np.sin(b) ** 2)


def cgamma(Z, out_unit='deg'):
    '''see Lilley 1998'''
    return catan2(Z[1, 1] + Z[0, 0], Z[0, 1] - Z[1, 0], out_unit)


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
    '''Complex arctan2 function'''
    real = np.arctan2(num.real, den.real)
    imag = np.arctan2(num.imag, den.imag)
    if out_unit == 'deg':
        real = real * 180 / np.pi
        imag = imag * 180 / np.pi
    return real + imag * 1j

def cgammas(Zs, out_unit='deg'):
    '''see Lilley 1998'''
    return np.array([cgamma(Z, out_unit) for Z in Zs])

lzdd = lambda z: z[1, 1] - z[0, 0]
lzos = lambda z: z[0, 1] + z[1, 0]
lzds = lambda z: z[1, 1] + z[0, 0]
lzod = lambda z: z[0, 1] - z[1, 0]

def theta_e(z, out_unit='deg'):
    '''Electric strike. See Lilley 1998.'''
    return 0.5 * (catan2(lzdd(z), lzos(z), out_unit) + catan2(lzds(z), lzod(z), out_unit))

def theta_h(z, out_unit='deg'):
    '''Magnetic strike. See Lilley 1998.'''
    return 0.5 * (catan2(lzdd(z), lzos(z), out_unit) - catan2(lzds(z), lzod(z), out_unit))

theta_es = lambda zs: np.array([theta_e(z) for z in zs])
theta_hs = lambda zs: np.array([theta_h(z) for z in zs])

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

def plot_mohr_imp(freqs, zs, axreal=None, aximag=None,
              cmap=plt.cm.jet_r, return_axes=False,
              fig=None, fign=None, clf=True):
    '''

    Args:

        - xlim, ylim: n x 2 complex ndarrays of x and y axes limits.
          Use None for automatic limits.

    '''
    if axreal is None and aximag is None:
        if fig is None:
            fig = plt.figure(fign)
        if clf:
            fig.clf()
        axreal = fig.add_subplot(121, aspect='equal')
        aximag = fig.add_subplot(122, aspect='equal', sharex=axreal, sharey=axreal)
    axreal.set_title('Real')
    aximag.set_title('Imag')
    angles = np.linspace(0, np.pi * 1.0, 50)
    for fi, freq in enumerate(freqs):
        z = zs[fi]
        zr = z.real
        zi = z.imag
        c = cmap(float(fi) / len(freqs))
        for ax, zp in zip((axreal, aximag), (zr, zi)):
            ax.plot([z12b(zp, a) for a in angles], [z11b(zp, a) for a in angles], color=c)
            ax.plot([lilley_Z4(zp), z12b(zp, 0)], [lilley_Z1(zp), z11b(zp, 0)], ls='-', color=c)
            ax.plot(lilley_Z4(zp), lilley_Z1(zp), marker='o', mfc=c, mec=c, markersize=2)
            ax.axvline(0, color='gray')
            ax.axhline(0, color='gray')
    if axreal.get_xlim()[0] > 0:
        axreal.set_xlim(0, None)
    if aximag.get_xlim()[0] > 0:
        aximag.set_xlim(0, None)
    legreal = axreal.legend(['%s Hz' % freqs[0], '%s Hz' % freqs[-1]], loc=2, numpoints=1)
    text_0, text_1 = legreal.get_texts()
    text_0.set_color(cmap(0))
    text_1.set_color(cmap(1 - 1e-10))
    if return_axes:
        return axreal, aximag


def plot_mohr_ptensor(freqs, ptensors, cmap=plt.cm.jet, ax=None, fig=None, fign=None, clf=True):
    if ax is None:
        if fig is None:
            fig = plt.figure(fign)
        if clf:
            fig.clf()
        ax = fig.add_subplot(111, aspect='equal')
    angles = np.linspace(0, np.pi * 1.0, 50)
    for fi, freq in enumerate(freqs):
        p = ptensors[fi]
        c = cmap(float(fi) / len(freqs))
        ax.plot([t11b(p, a) for a in angles], [t12b(p, a) for a in angles], color=c)
        ax.plot([L(p).t1, z11b(p, 0)], [L(p).t4, z12b(p, 0)], ls='-', color=c)
        ax.plot(L(p).t1, L(p).t4, marker='o', mfc=c, mec=c)
        ax.axvline(0, color='gray')
        ax.axhline(0, color='gray')
    legreal = ax.legend(['%s Hz' % freqs[0], '%s Hz' % freqs[-1]], loc=2, numpoints=1)
    text_0, text_1 = legreal.get_texts()
    text_0.set_color(cmap(0))
    text_1.set_color(cmap(1 - 1e-10))


def mrad2deg(arr):
    '''Convert milliradians to degrees, and keep it in the first quadrant.'''
    arr = arr / 1000 / np.pi * 180
    arr2 = np.empty_like(arr)
    for i, d in enumerate(arr):
        while d < -90:
            d += 180
        arr2[i] = d
    return arr2


def freq_lims(rd, names=None):
    '''From dictionary of sites, return min and max frequencies.'''
    f1 = 1e10
    f0 = 1e-10
    for name, sd in rd.items():
        if names:
            if not name in names:
                continue
        if 'freqs' in sd:
            if min(sd['freqs']) > f0:
                f0 = min(sd['freqs'])
            if max(sd['freqs']) < f1:
                f1 = max(sd['freqs'])
    return f0, f1


def rot_sd(sd, theta):
    '''Given dictionary of sites, rotate by theta degrees.'''
    sd2 = {}
    for name, s in sd.items():
        d = {'freqs': s['freqs'], 'zs': rot_arr(s['zs'], theta)}
        sd2[name] = site_data(d)
    return sd2

