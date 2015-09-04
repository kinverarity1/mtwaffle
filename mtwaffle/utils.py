import json

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
       if isinstance(o, np.ndarray):
            if np.iscomplex(o).any():
                return {'real': o.real.tolist(),
                        'imag': o.imag.tolist()}
            else:
                return o.tolist()
       return super(NumpyJSONEncoder, self).default(o)


def write_json(obj, fo, **kwargs):
    '''Write object *obj* (e.g. a dict) to file object *fo*.

    This is a wrapper for the standard library function :func:`json.dump``.
    It has a custom encoder for ndarrays, which are turned into (nested) lists.
    Complex ndarrays are turned into a dictionary with keys "real" and "imag"
    holding nested lists of the real and imaginary parts.

    See :func:`read_json`.

    '''
    kwargs['cls'] = NumpyJSONEncoder
    kwargs['encoding'] = 'ascii'
    kwargs['indent'] = 4
    return json.dump(obj, fo, **kwargs)


def resurrect_complex(d):
    for key, value in d.items():
        if isinstance(value, dict):
            if len(value.keys()) == 2 and 'real' in value and 'imag' in value:
                d[key] = np.asarray(value['real']) + np.asarray(value['imag']) * 1j
            else:
                resurrect_complex(value)
        elif isinstance(value, basestring):
            d[key] = value.encode('ascii')
        elif isinstance(value, list) or isinstance(value, tuple):
            d[key] = []
            for v in value:
                if isinstance(v, basestring):
                    d[key].append(v.encode('ascii'))
                else:
                    d[key].append(v)
        else:
            try:
                arr = np.asarray(value, dtype=np.float)
            except:
                print 'failed to convert', value
            else:
                d[key] = arr
            continue
    for key, value in d.items():
        if isinstance(value, list) or isinstance(value, tuple):
            all_floats = True
            for item in traverse(value):
                try:
                    f = np.float(item)
                except ValueError:
                    all_floats = False
                    break
            if all_floats:
                d[key] = np.asarray(value)


def traverse(o, tree_types=(list, tuple)):
    '''http://stackoverflow.com/a/6340578'''
    if isinstance(o, tree_types):
        for value in o:
            for subvalue in traverse(value):
                yield subvalue
    else:
        yield o


def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v)) 
        for k, v in dictionary.items())


def read_json(fo, **kwargs):
    '''Read JSON file (especially one written by :func:`write_json`).

    This reconstitutes ndarrays (including the decomposed complex ones), and
    converts all the Unicode strings back into ASCII.

    '''
    kwargs.setdefault('encoding', 'ascii')
    jsondict = json.load(fo, **kwargs)
    resurrect_complex(jsondict)
    return convert_keys_to_string(jsondict)