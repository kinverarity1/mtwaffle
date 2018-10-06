import inspect
import sys

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

    EXCLUDED_CALLABLES = ('between_freqs', )

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
    def phases(self):
        return self.phase_func(self.zs)

    def inspect_mt_callable(self, name):
        f = mt.callables[name]
        argnames = [     # Find arguments of callable from mtwaffle.mt
            p.name for p in inspect.signature(f).parameters.values()
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty
        ]
        return f, argnames

    def help(self, output=sys.stdout):
        '''Print a list of the attributes which are available.'''
        output.write('''
Attributes of mtwaffle.mtsite.Site are calculated using functions from the mtwaffle.mt module:

 mtsite.Site         mtwaffle.mt function
  attribute       (args are Site attributes)                  Function description
--------------  ------------------------------  ----------------------------------------------
''')
        label = lambda f: f.__doc__.splitlines()[0] if f.__doc__ else 'MISSING DOC'
        fnames = []
        for fname, f in mt.callables.items():
            try:
                getattr(self, fname)
            except:
                pass
            else:
                fnames.append(fname)
        for fname in fnames:
            f, argnames = self.inspect_mt_callable(fname)
            cname = self.__class__.__name__
            argsig = ', '.join(['{}'.format(arg) for arg in argnames])
            source = '{}({})'.format(fname, argsig)
            
            label_attr = '{}'.format(fname.ljust(14))
            label_source = source.ljust(30)
            label_help = label(f)
            output.write('{}  {}  {}\n'.format(label_attr, label_source, label_help))

            # print('{fname}({sig})'.format(
            #     fname=fname, sig=', '.join([
            #         '{c}.{a}'.format(c=self.__class__.__name__, a=arg) for arg in f_arg_names])))
            # output.write('{}.{}  --  {}\n'.format(
            #     self.__class__.__name__,
            #     fname.ljust(max([len(fi) for fi in fnames])), 
            #     doc(mt.callables[fname])
            #     )
            # )
    
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
        
        # See if we can complete a function from mtwaffle.mt using the
        # existing attributes in this Site:

        elif key in mt.callables and not key in self.EXCLUDED_CALLABLES:
            f, argnames = self.inspect_mt_callable(key)
            return f(*[getattr(self, arg) for arg in argnames])
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

    def plot_impedance_tensors(self, *args, **kwargs):
        return graphs.plot_impedance_tensors(
            self.zs, self.freqs, **kwargs)

    def plot_ptensell(self, *args, **kwargs):
        return graphs.plot_ptensell(
            self.ptensors, self.freqs, *args, **kwargs
        )

    def plot_ptensell_filled(self, *args, **kwargs):
        return graphs.plot_ptensell_filled(
            self.ptensors, self.freqs, *args, **kwargs
        )

    def plot_mohr_imp(self, *args, **kwargs):
        kwargs['title'] = kwargs.get('title', self.name)
        return graphs.plot_mohr_imp(
            self.zs, self.freqs, *args, **kwargs
        )

    def plot_mohr_ptensor(self, *args, **kwargs):
        return graphs.plot_mohr_ptensor(
            self.ptensors, self.freqs, *args, **kwargs
        )