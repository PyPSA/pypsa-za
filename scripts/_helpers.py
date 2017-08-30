import pandas as pd
import numpy as np
from six import iteritems

def madd(n, component, name=None, index=None, **kwargs):
    if index is None:
        index = pd.Index(kwargs.get('bus'))
    new_index = index
    if name is not None:
        new_index = new_index + (' ' + name)

    static = {}; series = {}
    for k, v in iteritems(kwargs):
        if isinstance(v, pd.Series) and isinstance(v.index, pd.DatetimeIndex):
            v = pd.DataFrame(np.repeat(v.values[:,np.newaxis], len(new_index), axis=1),
                             index=v.index, columns=new_index)
            series[k] = v
        elif isinstance(v, pd.DataFrame) and isinstance(v.index, pd.DatetimeIndex):
            v = pd.DataFrame(v)
            if len(v.columns.difference(index)) == 0:
                v = v.reindex(columns=index)
            v.columns = new_index
            series[k] = v
        elif isinstance(v, np.ndarray) and v.shape == (len(n.snapshots), len(new_index)):
            series[k] = pd.DataFrame(v, index=n.snapshots, columns=new_index)
        else:
            if isinstance(v, pd.Series):
                if len(v.index.difference(index)) == 0:
                    v = v.reindex(index)
                v = v.values
            static[k] = v

    n.import_components_from_dataframe(pd.DataFrame(static, index=new_index), component)
    for k, v in iteritems(series):
        n.import_series_from_dataframe(v, component, k)
    return new_index

def pdbcast(v, h):
    return pd.DataFrame(v.values.reshape((-1, 1)) * h.values,
                        index=v.index, columns=h.index)
