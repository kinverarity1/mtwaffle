import argparse
import glob
import os
import os.path as op
import sys
import time

try:
    from PyQt4 import QtGui, QtCore
except ImportError:
    from PySide import QtGui, QtCore
Qt = QtCore.Qt

import numpy as np
import pyqtgraph
from pyqtgraph import dockarea

import mt
import utils

def respfunc_viewer(path):
    app = QtGui.QApplication([])
    pyqtgraph.setConfigOption("background", "w")
    pyqtgraph.setConfigOption("foreground", "k")

    win = QtGui.QMainWindow()
    win.setWindowTitle("MT response function data viewer")

    darea = dockarea.DockArea()
    w = QtGui.QWidget()
    win.setCentralWidget(darea)

    taglist = QtGui.QListWidget(win)
    taglist.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
    taglist_dock = dockarea.Dock("Tags")
    taglist_dock.addWidget(taglist)
    darea.addDock(taglist_dock)

    sitelist = QtGui.QListWidget()
    sitelist.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
    sitelist_dock = dockarea.Dock("Tree...")
    sitelist_dock.addWidget(sitelist)
    darea.addDock(sitelist_dock, "left", taglist_dock)

    resplot = pyqtgraph.PlotWidget()
    resplot_dock = dockarea.Dock("APPARENT RESISTIVITY")
    resplot_dock.addWidget(resplot)
    darea.addDock(resplot_dock, "left", sitelist_dock)

    phaseplot = pyqtgraph.PlotWidget()
    phaseplot_dock = dockarea.Dock("PHASE")
    phaseplot_dock.addWidget(phaseplot)
    darea.addDock(phaseplot_dock, "bottom", resplot_dock)

    default_pen = [[(255,255,255,90)], dict(width=1)]
    select_pen = [["r"], dict(width=1.5)]
    skipflag_pen = [[(255,255,255,30)], dict(width=0.5)]

    resplotitem = resplot.getPlotItem()
    phaseplotitem = phaseplot.getPlotItem()
    resplotitem.invertX(True)
    phaseplotitem.invertX(True)
    resplotitem.setLogMode(x=True, y=True)
    phaseplotitem.setLogMode(x=True, y=False)
    phaseplotitem.vb.setXLink(resplotitem.vb)
    resplotitem.setYRange(np.log10(0.1), np.log10(1000))
    phaseplotitem.setYRange(0, 90)

    resvb = resplotitem.vb
    phasevb = phaseplotitem.vb

    data = utils.AttrDict()

    tagfns = glob.glob(op.join(path, "*-cal.json"))
    tag2fn = {}
    fn2tag = {}
    sites = set()
    tagfns.sort()
    
    data = utils.AttrDict()
    with open(op.join(path, "maskedfreqs.json"), mode="r") as f:
        maskedfreqs = utils.read_json(f)
    maskedlines = utils.AttrDict()
    datasymbols = utils.AttrDict()

    psymbols = utils.AttrDict({
        "xy": dict(pen=None, symbol="o", symbolBrush="b"),
        "yx": dict(pen=None, symbol="s", symbolBrush="r")
        })
    plines = utils.AttrDict({
        "xy": dict(pen="b"),
        "yx": dict(pen="r")
        })

    plotpens = utils.AttrDict({"xy": "b", "yx": "r",})
    plotsymbols = utils.AttrDict({"xy": "o", "yx": "s"})

    def plot(tag):

        if not hasattr(datasymbols[tag], "res_xy"):
            datasymbols[tag].res_xy = pyqtgraph.PlotDataItem(data[tag].freqs, data[tag].res_xy, **psymbols.xy)
            datasymbols[tag].res_yx = pyqtgraph.PlotDataItem(data[tag].freqs, data[tag].res_yx, **psymbols.yx)
            datasymbols[tag].phase_xy = pyqtgraph.PlotDataItem(data[tag].freqs, data[tag].phase_xy, **psymbols.xy)
            datasymbols[tag].phase_yx = pyqtgraph.PlotDataItem(data[tag].freqs, data[tag].phase_yx, **psymbols.yx)

            maskedlines[tag].res_xy = pyqtgraph.PlotDataItem(data[tag].freqs, data[tag].res_xy, **plines.xy)
            maskedlines[tag].res_yx = pyqtgraph.PlotDataItem(data[tag].freqs, data[tag].res_yx, **plines.yx)
            maskedlines[tag].phase_xy = pyqtgraph.PlotDataItem(data[tag].freqs, data[tag].phase_xy, **plines.xy)
            maskedlines[tag].phase_yx = pyqtgraph.PlotDataItem(data[tag].freqs, data[tag].phase_yx, **plines.yx)
            
            resplotitem.addItem(datasymbols[tag].res_xy)
            resplotitem.addItem(datasymbols[tag].res_yx)
            resplotitem.addItem(maskedlines[tag].res_xy)
            resplotitem.addItem(maskedlines[tag].res_yx)

            phaseplotitem.addItem(datasymbols[tag].phase_xy)
            phaseplotitem.addItem(datasymbols[tag].phase_yx)
            phaseplotitem.addItem(maskedlines[tag].phase_xy)
            phaseplotitem.addItem(maskedlines[tag].phase_yx)

        for i, freq in enumerate(data[tag].freqs):
            if maskedfreqs[tag]["masks"][i] == 0:
                data[tag].freqs[i] = float(maskedfreqs[tag]["freqs"][i])
            else:
                data[tag].freqs[i] = np.nan

        maskedlines[tag].res_xy.setData(data[tag].freqs, data[tag].res_xy)
        maskedlines[tag].res_yx.setData(data[tag].freqs, data[tag].res_yx)
        maskedlines[tag].phase_xy.setData(data[tag].freqs, data[tag].phase_xy)
        maskedlines[tag].phase_yx.setData(data[tag].freqs, data[tag].phase_yx)

    progress = QtGui.QProgressDialog("Loading data...", "Abort", 0, len(tagfns), win)
    progress.setWindowModality(QtCore.Qt.WindowModal)

    for i, tagfn in enumerate(tagfns):
        progress.setValue(i)
        tag = op.basename(tagfn).replace("-cal.json", "")
        tag2fn[tag] = tagfn
        fn2tag[tagfn] = tag
        site = tag.split("-")[0]
        sites.add(site)
        data[tag] = utils.read_json(tagfn)
        if not tag in maskedfreqs:
            maskedfreqs[tag] = utils.AttrDict({"freqs": data[tag].freqs.copy(), "masks": np.empty_like(data[tag].freqs) * 0})

        if not tag in maskedlines:
            maskedlines[tag] = utils.AttrDict()
            datasymbols[tag] = utils.AttrDict()

        plot(tag)

        if progress.wasCanceled():
            break

    progress.setValue(len(tagfns))

    resfreqselect = pyqtgraph.LinearRegionItem([0,-1])
    phasefreqselect = pyqtgraph.LinearRegionItem([0,-1])
    resplotitem.addItem(resfreqselect)
    phaseplotitem.addItem(phasefreqselect)

    def res_region_moved():
        phasefreqselect.setRegion(resfreqselect.getRegion())

    def phase_region_moved():
        resfreqselect.setRegion(phasefreqselect.getRegion())

    resfreqselect.sigRegionChanged.connect(res_region_moved)
    phasefreqselect.sigRegionChanged.connect(phase_region_moved)

    def populate_tag_list(filter_sites=None):
        if filter_sites:
            tags = [t for t in tag2fn.keys() if t.split("-")[0] in filter_sites]
        else:
            tags = sorted(tag2fn.keys())
        tags.sort()
        taglist.clear()
        for tag in tags:
            # print tag
            tagitem = QtGui.QListWidgetItem(taglist)
            tagitem.setText(tag)
        plot_per_tag_list()
        print

        
    def plot_per_tag_list():
        tags = [t.text() for t in taglist.selectedItems()]
        if not tags:
            tags = [t.text() for t in [taglist.item(i) for i in xrange(taglist.count())]]
        
        for plotitemtag, tagitems in datasymbols.items():
            if plotitemtag in tags:
                for item_name, item in tagitems.items():
                    item.setSymbol(plotsymbols[item_name[-2:]])
                    # item.setPen(None)#plotpens[item_name[-2:]])
            else:
                for item in tagitems.values():
                    item.setSymbol(None)
                    # item.setPen(None)

        for plotitemtag, tagitems in maskedlines.items():
            if plotitemtag in tags:
                for item_name, item in tagitems.items():
                    item.setPen(plotpens[item_name[-2:]])
            else:
                for item in tagitems.values():
                    item.setPen(None)

    def selected_site_names():
        return [s.text() for s in sitelist.selectedItems()]

    def pick_site():
        newsites = selected_site_names()
        populate_tag_list(newsites)
        # plot_per_tag_list()

    def toggle_selected_mask(value):
        tags = [str(t.text()) for t in taglist.selectedItems()]
        log_mask_range = resfreqselect.getRegion()
        fmin = 10 ** log_mask_range[0]
        fmax = 10 ** log_mask_range[1]
        for tag in tags:
            for i, freq in enumerate(maskedfreqs[tag]["freqs"]):
                if freq >= fmin and freq <= fmax:
                    maskedfreqs[tag]["masks"][i] = value
            plot(tag)
        print log_mask_range, tags, "\n"

    disable = QtGui.QPushButton("&Delete selected frequencies")
    enable = QtGui.QPushButton("&Enable selected frequencies")
    sitelist_dock.addWidget(disable)
    sitelist_dock.addWidget(enable)
    disable.clicked.connect(lambda: toggle_selected_mask(1))
    enable.clicked.connect(lambda: toggle_selected_mask(0))


    # def generate_key_press_event_handler(self, vb, event):
    #     vb.keyPressEvent(self, event)
    #     if event.key() is Qt.Key_X:
    #         toggle_selected_mask(mode="xy")
    #     elif event.key() is Qt.Key_Y:
    #         toggle_selected_mask(mode="yx")

    # resplotitem.vb.keyPressEvent = lambda 

    populate_tag_list()

    sites = sorted(list(sites))
    for site in sites:
        siteitem = QtGui.QListWidgetItem(sitelist)
        siteitem.setText(site)

    sitelist.itemSelectionChanged.connect(pick_site)
    taglist.itemSelectionChanged.connect(plot_per_tag_list)

    def cleanup():
        with open(op.join(path, "maskedfreqs.json"), mode="w") as f:
            utils.write_json(maskedfreqs, f)

    win.showMaximized()
    app.aboutToQuit.connect(cleanup)
    app.exec_()


def main():
    parser = argparse.ArgumentParser("MT response function data viewer")
    parser.add_argument("path")
    args = parser.parse_args(sys.argv[1:])
    return respfunc_viewer(args.path)


if __name__ == "__main__":
    main()
