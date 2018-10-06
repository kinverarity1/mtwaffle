import numpy as np
import attrdict

from mtwaffle import graphs
from mtwaffle import mt



class Site(attrdict.AttrDict):

    index_map = {
        'xx': [0, 0],
        'xy': [0, 1],
        'yx': [1, 0],
        'yy': [1, 1]
    }

    def __init__(self, freqs, zs, name='', phase_func=None, **kwargs):
        super(attrdict.AttrDict, self).__init__()
        self.freqs = np.asarray(freqs)
        self.zs = np.asarray(zs)
        self.name = name
        if phase_func is None:
            phase_func = mt.phase
        self.phase_func = phase_func
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def periods(self):
        return 1. / self.freqs

    @property
    def appres(self):
        return mt.appres(self.zs, self.freqs)

    @property
    def phases(self):
        return self.phase_func(self.zs)
    
    def get_property(self, key):
        # Is the key ending with xx, xy, yx, or yy?
        if key[-2:] in self.index_map:
            indices = self.index_map[key[-2:]]
            if key.startswith('res_'):
                return self.appres[[Ellipsis] + indices]
            elif key.startswith('phase_'):
                return self.phases[[Ellipsis] + indices]
            elif key.startswith('zr_'):
                return self.zs.real[[Ellipsis] + indices]
            elif key.startswith('zi_'):
                return self.zs.imag[[Ellipsis] + indices]
        else:
            if key == 'ptensazimuths':
                return mt.ptensazimuths(self.zs)
            if key == 'ptensors':
                return mt.ptensors(self.zs)
            if key == 'normptskew':
                return mt.normptskew(self.zs)
        return False

    def __getattr__(self, key):
        value = self.get_property(key)
        if value is False:
            return super(attrdict.AttrDict, self).__getattr__(key)
        else:
            return value

    def __getitem__(self, key):
        value = self.get_property(key)
        if value is False:
            return super(attrdict.AttrDict, self).__getitem__(key)
        else:
            return value

    def plot_res_phase(self, **kwargs):
        args = (
            (self.freqs, self.freqs),
            (self.res_xy, self.res_yx),
            (self.phase_xy, self.phase_yx),
        )
        if not 'res_indiv_kws' in kwargs:
            kwargs['res_indiv_kws'] = (
                {'label': 'xy', 'color': 'b'},
                {'label': 'yx', 'color': 'g'},
            )
        return graphs.plot_res_phase(*args, **kwargs)

    def plot_impedance_tensors(self, **kwargs):
        return graphs.plot_impedance_tensors(
            self.zs, self.freqs, **kwargs)

    def plot_ptensell(self, **kwargs):
        return graphs.plot_ptensell(
            self.ptens, self.freqs, **kwargs
        )

    def plot_ptensell_filled(self, **kwargs):
        return graphs.plot_ptensell_filled(
            self.ptens, self.freqs, **kwargs
        )

    def plot_mohr_imp(self, **kwargs):
        kwargs['title'] = kwargs.get('title', self.name)
        return graphs.plot_mohr_imp(
            self.zs, self.freqs, **kwargs
        )

    def plot_mohr_ptensor(self, **kwargs):
        return graphs.plot_mohr_ptensor(
            self.ptens, self.freqs, **kwargs
        )