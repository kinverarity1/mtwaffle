import argparse
import glob
import os
import os.path as op
import sys
import time

try:
    from PySide import QtGui, QtCore
except ImportError:
    from PyQt4 import QtGui, QtCore

import numpy as np
import pyqtgraph
from pyqtgraph import dockarea

import mt
import utils

def respfunc_viewer(path):
    app = QtGui.QApplication([])
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
    maskdata = utils.AttrDict()

    maskedlines = utils.AttrDict()
    datasymbols = utils.AttrDict()

    progress = QtGui.QProgressDialog("Loading data...", "Abort", 0, len(tagfns), win)
    progress.setWindowModality(QtCore.Qt.WindowModal)

    plotpens = utils.AttrDict({"xy": "b", "yx": "r"})
    plotsymbols = utils.AttrDict({"xy": "o", "yx": "s"})

    for i, tagfn in enumerate(tagfns):
        progress.setValue(i)
        tag = op.basename(tagfn).replace("-cal.json", "")
        tag2fn[tag] = tagfn
        fn2tag[tagfn] = tag        
        site = tag.split("-")[0]
        sites.add(site)
        data[tag] = utils.read_json(tagfn)
        maskdata[tag] = utils.read_json(tagfn)
        if not tag in maskedlines:
            maskedlines[tag] = utils.AttrDict()
            datasymbols[tag] = utils.AttrDict()

        datasymbols[tag].res_xy = resplot.plot(data[tag].freqs, data[tag].res_xy, pen=None, symbol=plotsymbols.xy)
        datasymbols[tag].res_yx = resplot.plot(data[tag].freqs, data[tag].res_yx, pen=None, symbol=plotsymbols.yx)
        datasymbols[tag].phase_xy = phaseplot.plot(data[tag].freqs, data[tag].phase_xy, pen=None, symbol=plotsymbols.xy)
        datasymbols[tag].phase_yx = phaseplot.plot(data[tag].freqs, data[tag].phase_yx, pen=None, symbol=plotsymbols.yx)

        maskedlines[tag].res_xy = resplot.plot(data[tag].freqs, data[tag].res_xy, pen=plotpens.xy)
        maskedlines[tag].res_yx = resplot.plot(data[tag].freqs, data[tag].res_yx, pen=plotpens.yx)
        maskedlines[tag].phase_xy = phaseplot.plot(data[tag].freqs, data[tag].phase_xy, pen=plotpens.xy)
        maskedlines[tag].phase_yx = phaseplot.plot(data[tag].freqs, data[tag].phase_yx, pen=plotpens.yx)

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
        taglist.clear()
        for tag in tags:
            tagitem = QtGui.QListWidgetItem(taglist)
            tagitem.setText(tag)
        plot_per_tag_list()

        
    def plot_per_tag_list():
        tags = [t.text() for t in taglist.selectedItems()]
        if not tags:
            tags = [t.text() for t in [taglist.item(i) for i in xrange(taglist.count())]]
        # for plotitemtag, tagitems in maskedlines.items():
        #     if plotitemtag in tags:
        #         for item_name, item in tagitems.items():
        #             item.setPen(plotpens[item_name[-2:]])
        #     else:
        #         for item in tagitems.values():
        #             item.setPen(None)
        for plotitemtag, tagitems in datasymbols.items():
            if plotitemtag in tags:
                for item_name, item in tagitems.items():
                    item.setSymbol(plotsymbols[item_name[-2:]])
                    item.setPen(None)#plotpens[item_name[-2:]])
            else:
                for item in tagitems.values():
                    item.setSymbol(None)
                    item.setPen(None)

    def selected_site_names():
        return [s.text() for s in sitelist.selectedItems()]

    def pick_site():
        newsites = selected_site_names()
        populate_tag_list(newsites)
        # plot_per_tag_list()

    populate_tag_list()

    sites = sorted(list(sites))
    for site in sites:
        siteitem = QtGui.QListWidgetItem(sitelist)
        siteitem.setText(site)

    sitelist.itemSelectionChanged.connect(pick_site)
    taglist.itemSelectionChanged.connect(plot_per_tag_list)

    win.showMaximized()
    app.exec_()


def main():
    parser = argparse.ArgumentParser("MT response function data viewer")
    parser.add_argument("path")
    args = parser.parse_args(sys.argv[1:])
    return respfunc_viewer(args.path)


if __name__ == "__main__":
    main()
