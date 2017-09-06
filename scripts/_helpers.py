import pandas as pd
import numpy as np
from six import iteritems

import pypsa

def madd(n, component, name=None, index=None, **kwargs):
    if index is None:
        index = pd.Index(kwargs.get('bus'))
    new_index = index
    if name is not None:
        new_index = new_index + (' ' + name)

    static = {}; series = {}
    for k, v in iteritems(kwargs):
        if isinstance(v, pd.Series):
            if len(v.index.intersection(index)) > 0:
                static[k] = v.reindex(index).values
            else:
                v = pd.DataFrame(np.repeat(v.values[:,np.newaxis], len(new_index), axis=1),
                                index=v.index, columns=new_index)
                series[k] = v
        elif isinstance(v, pd.DataFrame):
            v = pd.DataFrame(v).reindex(columns=index)
            v.columns = new_index
            series[k] = v
        elif isinstance(v, np.ndarray) and v.shape == (len(n.snapshots), len(new_index)):
            series[k] = pd.DataFrame(v, index=n.snapshots, columns=new_index)
        else:
            static[k] = v

    n.import_components_from_dataframe(pd.DataFrame(static, index=new_index), component)
    for k, v in iteritems(series):
        n.import_series_from_dataframe(v, component, k)
    return new_index

def pdbcast(v, h):
    return pd.DataFrame(v.values.reshape((-1, 1)) * h.values,
                        index=v.index, columns=h.index)


def load_network(fn, opts, combine_hydro_ps=True):
    n = pypsa.Network(fn)

    n.loads["carrier"] = n.loads.bus.map(n.buses.carrier) + " load"
    n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)

    n.links["carrier"] = (n.links.bus0.map(n.buses.carrier) + "-" + n.links.bus1.map(n.buses.carrier))
    n.lines["carrier"] = "AC line"
    n.transformers["carrier"] = "AC transformer"

    if combine_hydro_ps:
        n.storage_units.loc[n.storage_units.carrier.isin({'Pumped storage', 'Hydro'}), 'carrier'] = 'Hydro+PS'

    # #if the carrier was not set on the heat storage units
    # bus_carrier = n.storage_units.bus.map(n.buses.carrier)
    # n.storage_units.loc[bus_carrier == "heat","carrier"] = "water tanks"

    for name in opts['heat_links'] + opts['heat_generators']:
        n.links.loc[n.links.index.to_series().str.endswith(name), "carrier"] = name

    return n
