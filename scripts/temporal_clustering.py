#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:00:41 2021

aggregate PyPSA network to representative periods

@author: bw0928
"""
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
import tsam.timeseriesaggregation as tsam

def prepare_timeseries_tsam(network, normed=False):
    """
    """
    p_max_pu = network.generators_t.p_max_pu 
    load = network.loads_t.p_set
    inflow = network.storage_units_t.inflow 
    years=network.snapshots.get_level_values(0).unique()
    p_max_pu_max = network.generators_t.p_max_pu.groupby(level=0).max() 
    load_max = network.loads_t.p_set.groupby(level=0).max() 
    inflow_max = network.storage_units_t.inflow.groupby(level=0).max() 

    if normed:
        load = load / load_max
        p_max_pu = p_max_pu / p_max_pu_max
        inflow = inflow / inflow_max

    df = pd.concat([load,p_max_pu,inflow], axis=1)

    df_max = pd.DataFrame(0,index=network.investment_periods,columns=df.columns)
    df_max[load_max.columns] = load_max
    df_max[p_max_pu_max.columns] = p_max_pu_max
    df_max[inflow_max.columns] = inflow_max

    df=df.fillna(0)

    return df, df_max   


def cluster_snapshots(network, normed=False, noTypicalPeriods=30, extremePeriodMethod = 'None',
                      rescaleClusterPeriods=False, hoursPerPeriod=24, clusterMethod='hierarchical',
                      solver='cbc',predefClusterOrder=None):
    # Function modified from code by Lisa Zeyen under https://github.com/lisazeyen/learning_curve

    timeseries_df, timeseries_df_max=prepare_timeseries_tsam(network,True)
    new_snapshots={}
    map_snapshots_to_periods={}

    for y in network.snapshots.get_level_values(0).unique():
        new_snapshots[y], map_snapshots_to_periods[y] = tsam_clustering(timeseries_df.loc[y], timeseries_df_max.loc[y,:], 
                            noTypicalPeriods=noTypicalPeriods, extremePeriodMethod = extremePeriodMethod,
                            rescaleClusterPeriods=rescaleClusterPeriods, hoursPerPeriod=hoursPerPeriod, clusterMethod=clusterMethod,
                            solver=solver,predefClusterOrder=predefClusterOrder)

        if y == network.snapshots.get_level_values(0).unique()[0]:
            snapshots_stack = new_snapshots[y]
            map_snapshots_to_periods_stack = map_snapshots_to_periods[y]
        else:
            snapshots_stack=snapshots_stack.append(new_snapshots[y])
            map_snapshots_to_periods_stack=pd.concat([map_snapshots_to_periods_stack,map_snapshots_to_periods[y]])

    snapshots_clustered = pd.MultiIndex.from_arrays([snapshots_stack.index.year, snapshots_stack.index])
    weightings = snapshots_stack.weightings
    weightings.index=snapshots_clustered
    cluster_map = map_snapshots_to_periods_stack

    network.cluster = cluster_map
    # set new snapshots
    network.set_snapshots(snapshots_clustered)
    network.snapshot_weightings = network.snapshot_weightings.mul(weightings, axis=0)

    return network

def tsam_clustering(timeseries_df,  timeseries_df_max, noTypicalPeriods=30, extremePeriodMethod = 'None', 
                    rescaleClusterPeriods=False, hoursPerPeriod=24, clusterMethod='hierarchical',
                    solver='cbc',predefClusterOrder=None):
    # Function developed by Lisa Zeyen under https://github.com/lisazeyen/learning_curve

    aggregation = tsam.TimeSeriesAggregation(timeseries_df, noTypicalPeriods=noTypicalPeriods, extremePeriodMethod = extremePeriodMethod, 
                                            rescaleClusterPeriods=rescaleClusterPeriods, hoursPerPeriod=hoursPerPeriod, clusterMethod=clusterMethod,
                                            solver=solver, predefClusterOrder=predefClusterOrder)

    clustered = aggregation.createTypicalPeriods()
    clustered = clustered.mul(timeseries_df_max)
    map_snapshots_to_periods = aggregation.indexMatching()
    map_snapshots_to_periods["day_of_year"] = map_snapshots_to_periods.index.day_of_year
    cluster_weights = aggregation.clusterPeriodNoOccur
    clusterCenterIndices = aggregation.clusterCenterIndices
    mapped_day_of_year = map_snapshots_to_periods.day_of_year-1
    if map_snapshots_to_periods.index[0].is_leap_year:
        mapped_day_of_year.loc[mapped_day_of_year>59]-=1    

    new_snapshots = map_snapshots_to_periods[mapped_day_of_year.isin(clusterCenterIndices)]
    new_snapshots["weightings"] = new_snapshots["PeriodNum"].map(cluster_weights).astype(float)
    clustered.set_index(new_snapshots.index, inplace=True)

    # last hour of typical period
    last_hour = new_snapshots[new_snapshots["TimeStep"]==hoursPerPeriod-1]
    # first hour
    first_hour = new_snapshots[new_snapshots["TimeStep"]==0]

    # add typical period name and last hour to mapping original snapshot-> typical
    map_snapshots_to_periods["RepresentativeDay"] = map_snapshots_to_periods["PeriodNum"].map(last_hour.set_index(["PeriodNum"])["day_of_year"].to_dict())
    map_snapshots_to_periods["last_hour_RepresentativeDay"] = map_snapshots_to_periods["PeriodNum"].map(last_hour.reset_index().set_index(["PeriodNum"])["timestep"].to_dict())
    map_snapshots_to_periods["first_hour_RepresentativeDay"] = map_snapshots_to_periods["PeriodNum"].map(first_hour.reset_index().set_index(["PeriodNum"])["timestep"].to_dict())

    return new_snapshots, map_snapshots_to_periods


def apply_time_segmentation(n, segments):
    logger.info(f"Aggregating time series to {segments} segments.")
    try:
        import tsam.timeseriesaggregation as tsam
    except:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
        )

    multi_segmented = []
    multi_weighted = []
    multi_snahots = []
    multi_p_max_pu = []
    multi_p_min_pu = []
    multi_load = []
    multi_inflow =[]

    # Time segmentation is applied internally to each investment period
    for y in n.investment_periods:
        p_max_pu_norm = n.generators_t.p_max_pu.loc[y].max()
        p_min_pu_norm = n.generators_t.p_min_pu.loc[y].min()
        p_max_pu = n.generators_t.p_max_pu.loc[y] / p_max_pu_norm
        p_min_pu = n.generators_t.p_min_pu.loc[y] / p_min_pu_norm
        
        load_norm = n.loads_t.p_set.loc[y].max()
        load = n.loads_t.p_set.loc[y] / load_norm

        inflow_norm = n.storage_units_t.inflow.loc[y].max()
        inflow = n.storage_units_t.inflow.loc[y] / inflow_norm

        raw = pd.concat([p_max_pu, p_min_pu, load, inflow], axis=1, sort=False)

        solver_name = snakemake.config["solving"]["solver"]["name"]
        extremePeriodMethod = snakemake.config["tsam_clustering"]["extremePeriodMethod"]
        agg = tsam.TimeSeriesAggregation(
            raw,
            hoursPerPeriod=len(raw),
            noTypicalPeriods=1,
            noSegments=int(segments),
            segmentation=True,
            extremePeriodMethod = extremePeriodMethod
            solver=solver_name,
        )

        segmented = agg.createTypicalPeriods()
        weightings = segmented.index.get_level_values("Segment Duration")
        offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
        snapshots = [n.snapshots.loc[y][0] + pd.Timedelta(f"{offset}h") for offset in offsets]

        # Append segmented timeseries data for each investment
        multi_segmented = pd.concat([multi_segmented, segmented])
        multi_weightings = pd.concat([multi_weightings, weightings])
        multi_p_max_pu = pd.concat([multi_p_max_pu, segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm])
        multi_p_min_pu = pd.concat([multi_p_min_pu, segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm])
        multi_load = pd.concat([multi_load, segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm])
        multi_inflow = pd.concat([multi_inflow, segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm])

        multi_snapshots = pd.concat([multi_snapshots, snapshots])        
    
    multi_snapshots = pd.MultiIndex.from_arrays([multi_snapshots.years, multi_snapshots]) # recreate multi index

    n.set_snapshots(pd.DatetimeIndex(multi_snapshots, name="name"))
    n.snapshot_weightings = pd.Series(
        weightings, index=multi_snapshots, name="weightings", dtype="float64"
    )

    multi_segmented.index = snapshots
    n.generators_t.p_max_pu = multi_p_max_pu[n.generators_t.p_max_pu.columns]
    n.generators_t.p_min_pu = multi_p_min_pu[n.generators_t.p_min_pu.columns]
    n.loads_t.p_set = multi_loads[n.loads_t.p_set.columns]
    n.storage_units_t.inflow = multi_inflow[n.storage_units_t.inflow.columns]

    return n



