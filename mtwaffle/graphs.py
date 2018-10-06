import logging

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from mtwaffle import mt


logger = logging.getLogger(__name__)



def plot_res_phase2(freqs, Zs, phase_func=None, **kwargs):
    """Quick wrapper for plotting the two modes resistivity and phase.

    Args:
        - *freqs*: n x 1 ndarray
        - *Zs*: n x 2 x 2 complex ndarray
        - *phase_func*: function used to calculate phase

    Kwargs: passed to :func:`plot_res_phase`

    """
    res = mt.appres(freqs, Zs)
    if phase_func is None:
        from mtwaffle import mt
        phase_func = mt.phase
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
        alpha = mt.ptens_alpha(P)
        beta = mt.ptens_beta(P)

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


def plot_mohr_imp(freqs, zs, axreal=None, aximag=None,
              cmap=plt.cm.jet_r,
              fig=None, fign=None, clf=True, title=None):
    '''Plot Mohr circles

    Args:
        freqs (n x 1 ndarray): list of frequencies
        zs (n x 2 x 2 complex ndarray): list of impedance tensors
        axreal (matplotlib Axes): axes for the LHS plot of Re(Z) (optional)
        aximag (matplotlib Axes): axes for the RHS plot of Im(Z) (optional)
        cmap (matplotlib colormap): (optional)
        fig (matplotlib Figure): (optional)
        fign (int): matplotlib figure number (optional)
        clf (bool): clear existing matplotlib figure (optional)
        title (str): figure title (optional)

    Returns: (axreal, aximag): tuple of the matplotlib Axes

    '''
    if axreal is None and aximag is None:
        if fig is None:
            fig = plt.figure(fign)
        if clf:
            fig.clf()
        axreal = fig.add_subplot(121, aspect='equal')
        aximag = fig.add_subplot(122, aspect='equal', sharex=axreal, sharey=axreal)
        if title:
            fig.suptitle(title)
    axreal.set_title('Real')
    aximag.set_title('Imag')
    angles = np.linspace(0, np.pi * 1.0, 50)
    freq_0_line = None
    for fi, freq in enumerate(freqs):
        z = zs[fi]
        zr = z.real
        zi = z.imag
        c = cmap(float(fi) / len(freqs))
        for ax, zp in zip((axreal, aximag), (zr, zi)):
            line, = ax.plot([mt.z12b(zp, a) for a in angles], [mt.z11b(zp, a) for a in angles], color=c)
            ax.plot([mt.lilley_Z4(zp), mt.z12b(zp, 0)], [mt.lilley_Z1(zp), mt.z11b(zp, 0)], ls='-', color=c)
            ax.plot(mt.lilley_Z4(zp), mt.lilley_Z1(zp), marker='o', mfc=c, mec=c, markersize=2)
            ax.axvline(0, color='gray')
            ax.axhline(0, color='gray')
        if fi == 0:
            freq_0_line = line
    if axreal.get_xlim()[0] > 0:
        axreal.set_xlim(0, None)
    if aximag.get_xlim()[0] > 0:
        aximag.set_xlim(0, None)
    legreal = axreal.legend((freq_0_line, line), ['%s Hz' % freqs[0], '%s Hz' % freqs[-1]], loc=2, numpoints=2)
    text_0, text_1 = legreal.get_texts()
    text_0.set_color(cmap(0))
    text_1.set_color(cmap(1 - 1e-10))
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