#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:00:41 2021

aggregate PyPSA network to representative periods

@author: bw0928
"""
import pandas as pd
import logging
logger = logging.getLogger(__name__)
import tsam.timeseriesaggregation as tsam

def prepare_timeseries_tsam(network, normed=False):
    """
    """
    p_max_pu = network.generators_t.p_max_pu 
    load = network.loads_t.p_set
    inflow = network.storage_units_t.inflow 

    if isinstance(network.snapshots, pd.MultiIndex):
        years=network.snapshots.get_level_values(0).unique()
        p_max_pu_max = network.generators_t.p_max_pu.groupby(level=0).max() 
        load_max = network.loads_t.p_set.groupby(level=0).max() 
        inflow_max = network.storage_units_t.inflow.groupby(level=0).max() 
    else:
        p_max_pu_max = network.generators_t.p_max_pu.max() 
        load_max = network.loads_t.p_set.max() 
        inflow_max = network.storage_units_t.inflow.max() 

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

    timeseries_df, timeseries_df_max=prepare_timeseries_tsam(network,True)
    # If modelling a single year 
    if isinstance(network.snapshots, pd.MultiIndex):
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

    # If modelling multiple years cluster in each year and then stack the results back to a MultiIndex
    else:
        new_snapshots, map_snapshots_to_periods = tsam_clustering(n, normed=normed, 
                            noTypicalPeriods=noTypicalPeriods, extremePeriodMethod = extremePeriodMethod,
                            rescaleClusterPeriods=rescaleClusterPeriods, hoursPerPeriod=hoursPerPeriod, clusterMethod=clusterMethod,
                            solver=solver,predefClusterOrder=predefClusterOrder)
        
        snapshots_clustered = new_snapshots.index
        weightings = new_snapshots.weightings
        weightings.index=snapshots_clustered
        cluster_map = map_snapshots_to_periods

    network.cluster = cluster_map
    # set new snapshots
    network.set_snapshots(snapshots_clustered)
    network.snapshot_weightings = network.snapshot_weightings.mul(weightings, axis=0)

    return network

def tsam_clustering(timeseries_df,  timeseries_df_max, noTypicalPeriods=30, extremePeriodMethod = 'None', 
                    rescaleClusterPeriods=False, hoursPerPeriod=24, clusterMethod='hierarchical',
                    solver='cbc',predefClusterOrder=None):

    aggregation = tsam.TimeSeriesAggregation(timeseries_df, noTypicalPeriods=noTypicalPeriods, extremePeriodMethod = extremePeriodMethod, 
                                            rescaleClusterPeriods=rescaleClusterPeriods, hoursPerPeriod=hoursPerPeriod, clusterMethod=clusterMethod,
                                            solver=solver, predefClusterOrder=predefClusterOrder)

    clustered = aggregation.createTypicalPeriods()
    clustered = clustered.mul(timeseries_df_max)
    map_snapshots_to_periods = aggregation.indexMatching()
    map_snapshots_to_periods["day_of_year"] = map_snapshots_to_periods.index.day_of_year
    cluster_weights = aggregation.clusterPeriodNoOccur
    clusterCenterIndices = aggregation.clusterCenterIndices
    new_snapshots = map_snapshots_to_periods[(map_snapshots_to_periods.day_of_year-1).isin(clusterCenterIndices)]
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



