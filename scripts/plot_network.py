if 'snakemake' not in globals():
    from vresutils import Dict
    from snakemake.rules import expand
    import yaml
    snakemake = Dict()
    snakemake.wildcards = Dict(#cost=#'IRP2016-Apr2016',
                               cost='CSIR-Expected-Apr2016',
                               mask='redz',
                               sectors='E',
                               opts='Co2L',
                               attr='p_nom')
    snakemake.input = Dict(network='../results/networks/{cost}_{mask}_{sectors}_{opts}'.format(**snakemake.wildcards),
                           supply_regions='../data/external/supply_regions/supply_regions.shp',
                           maskshape = "../data/external/masks/{mask}".format(**snakemake.wildcards))
    snakemake.output = (expand('../results/plots/network_{cost}_{mask}_{sectors}_{opts}_{attr}.pdf',
                              **snakemake.wildcards) + 
                        expand('../results/plots/network_{cost}_{mask}_{sectors}_{opts}_{attr}_ext.pdf',
                              **snakemake.wildcards))
    with open('../config.yaml') as f:
        snakemake.config = yaml.load(f)
else:
    import matplotlib as mpl
    mpl.use('Agg')

from _helpers import load_network, aggregate_p, aggregate_costs
from vresutils import plot as vplot


import os
import pypsa
import pandas as pd
import geopandas as gpd
import numpy as np
from itertools import product, chain
from six.moves import map, zip
from six import itervalues, iterkeys
from collections import OrderedDict as odict

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
to_rgba = mpl.colors.colorConverter.to_rgba

def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()
    def axes2pt():
        return np.diff(ax.transData.transform([(0,0), (1,1)]), axis=0)[0] * (72./fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses: e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0,0), radius=(s/scale)**0.5, **kw) for s in sizes]

import seaborn as sns
plt.style.use(['classic', 'seaborn-white',
               {'axes.grid': False, 'grid.linestyle': '--', 'grid.color': u'0.6',
                'patch.linewidth': 0.5,
                #'font.size': 10,
                'lines.linewidth': 1.5
               }])

opts = snakemake.config['plotting']
map_figsize = opts['map']['figsize']
map_boundaries = opts['map']['boundaries']

n = load_network(snakemake.input.network, opts)
supply_regions = gpd.read_file(snakemake.input.supply_regions).buffer(-0.005) #.to_crs(n.crs)
renewable_regions = gpd.read_file(snakemake.input.maskshape).to_crs(supply_regions.crs)

## DATA
line_colors = {'Line': to_rgba("r", 0.7),
               'Link': to_rgba("purple", 0.7)}
tech_colors = opts['tech_colors']

if snakemake.wildcards.attr == 'p_nom':
    # bus_sizes = n.generators_t.p.sum().loc[n.generators.carrier == "load"].groupby(n.generators.bus).sum()
    bus_sizes = pd.concat((n.generators.query('carrier != "load"').groupby(['bus', 'carrier']).p_nom_opt.sum(),
                           n.storage_units.groupby(['bus', 'carrier']).p_nom_opt.sum()))
    line_widths = pd.concat(dict(Line=n.lines.s_nom_opt - n.lines.s_nom,
                                 Link=n.links.p_nom_opt - n.links.p_nom))
else:
    raise 'plotting of {} has not been implemented yet'.format(plot)


line_colors_with_alpha = \
pd.concat(dict(Line=((n.lines.s_nom_opt - n.lines.s_nom) / n.lines.s_nom > 1e-3)
                    .map({True: line_colors['Line'], False: to_rgba(line_colors['Line'], 0.)}),
               Link=((n.links.p_nom_opt - n.links.p_nom) / n.links.p_nom > 1e-3)
                    .map({True: line_colors['Link'], False: to_rgba(line_colors['Link'], 0.)})))

## FORMAT
linewidth_factor = opts['map'][snakemake.wildcards.attr]['linewidth_factor']
bus_size_factor  = opts['map'][snakemake.wildcards.attr]['bus_size_factor']

## PLOT
fig, ax = plt.subplots(figsize=map_figsize)
vplot.shapes(supply_regions.geometry, colour='k', outline='k', ax=ax, rasterized=True)
vplot.shapes(renewable_regions.geometry, colour='gray', alpha=0.2, ax=ax, rasterized=True)
n.plot(line_widths=line_widths/linewidth_factor,
       line_colors=line_colors_with_alpha,
       bus_sizes=bus_sizes/bus_size_factor,
       bus_colors=tech_colors,
       boundaries=map_boundaries,
       basemap=True,
       ax=ax)
ax.set_aspect('equal')
ax.axis('off')

x1, y1, x2, y2 = map_boundaries
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)


# Rasterize basemap
#for c in ax.collections[:2]: c.set_rasterized(True)

# LEGEND
handles = []
labels = []
for s in (10, 5, 1):
    handles.append(plt.Line2D([0],[0],color=line_colors['Line'],
                              linewidth=s*1e3/linewidth_factor))
    labels.append("{} GW".format(s))
l1 = ax.legend(handles, labels,
               loc="upper left", bbox_to_anchor=(0.25, 1.),
               frameon=False,
               labelspacing=0.8, handletextpad=1.5, fontsize=10,
               title='Transmission')
ax.add_artist(l1)

handles = make_legend_circles_for([10e3, 5e3, 1e3], scale=bus_size_factor, facecolor="w")
labels = ["{} GW".format(s) for s in (10, 5, 3)]
l2 = ax.legend(handles, labels,
               loc="upper left",
               frameon=False, fontsize=10, labelspacing=1.0,
               title='Generation',#fontsize='small',
               handler_map=make_handler_map_to_scale_circles_as_in(ax))
ax.add_artist(l2)

techs = pd.Index(opts['vre_techs'] + opts['conv_techs'] + opts['storage_techs']) & (bus_sizes.index.levels[1])
handles = []
labels = []
for t in techs:
    handles.append(plt.Line2D([0], [0], color=tech_colors[t], marker='o', markersize=8, linewidth=0))
    labels.append(opts['nice_names'].get(t, t))
l3 = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.74, 0.0),
          fontsize=10, handletextpad=0., columnspacing=0.5, ncol=2, title='Technology')


fig.savefig(snakemake.output[0], dpi=150, bbox_inches='tight', bbox_extra_artists=[l1,l2,l3])

n = load_network(snakemake.input.network, opts, combine_hydro_ps=False)

## Add total energy p

ax1 = ax = fig.add_axes([-0.15, 0.555, 0.2, 0.2])
ax.set_title('Energy per technology', dict(fontsize=12))

e_primary = aggregate_p(n).loc[lambda s: s>0].drop('load')

patches, texts, autotexts = ax.pie(e_primary,
       startangle=90,
       labels = e_primary.rename(opts['nice_names_n']).index,
      autopct='%.0f%%',
      shadow=False,
          colors = [tech_colors[tech] for tech in e_primary.index])
for t1, t2, i in zip(texts, autotexts, e_primary.index):
    if e_primary.at[i] < 0.02 * e_primary.sum():
        t1.remove()
        t2.remove()
    else:
        t1.set_fontsize(12)
        t2.set_fontsize(12)
        t2.set_color('k')
        t2.set_weight('bold')

## Add average system cost bar plot
ax2 = ax = fig.add_axes([-0.1, 0.2, 0.1, 0.33])
total_load = n.loads_t.p.sum().sum()
costs = aggregate_costs(n).reset_index(level=0, drop=True)
costs = costs['capital'].add(costs['marginal'], fill_value=0.)

costs_graph = pd.DataFrame(dict(a=costs.drop('load')),
                          index=['AC line', 'Wind', 'PV', 'Nuclear',
                                 'Coal', 'OCGT', 'CAES', 'Battery'])
bottom = np.array([0.])
texts = []

for i,ind in enumerate(costs_graph.index):
    data = np.asarray(costs_graph.loc[ind])/total_load
    ax.bar([0.1], data, bottom=bottom, color=tech_colors[ind], width=0.8, zorder=-1)
    bottom = bottom+data

    if abs(data[-1]) < 10:
        continue
    text = ax.text(1.1,(bottom-0.5*data)[-1]-15,opts['nice_names'].get(ind,ind),size=12)
    texts.append(text)

ax.set_ylabel("Average system cost [EUR/MWh]")
ax.set_xlim([0,1])
ax.set_xticks([])
ax.set_xticklabels([])
ax.grid(True, axis="y", color='k', linestyle='dotted')

#fig.tight_layout()

fig.savefig(snakemake.output[1], transparent=True, bbox_inches='tight', bbox_extra_artists=[l1, l2, l3, ax1, ax2])


# if False:
#     filename = "total-pie-{}".format(key).replace(".","-")+".pdf"
#     print("Saved to {}".format(filename))
#     fig.savefig(filename,transparent=True,bbox_inches='tight',bbox_extra_artists=texts)

# #ax.set_title('Expansion to 1.25 x today\'s line volume at 256 clusters')f True or 'snakemake' not in globals():

