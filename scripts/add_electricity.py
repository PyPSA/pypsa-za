# coding: utf-8
import logging
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
import xarray as xr
from _helpers import configure_logging, getContinent, update_p_nom_max, pdbcast
from shapely.validation import make_valid
from shapely.geometry import Point
from vresutils import transfer as vtransfer
idx = pd.IndexSlice
logger = logging.getLogger(__name__)

def normed(s):
    return s / s.sum()

def remove_leap_day(df):
    return df[~((df.index.month == 2) & (df.index.day == 29))]

def map_generator_parameters():

    ps_f = dict(PHS_efficiency="Pump Efficiency (%)",
                PHS_units="Pump Units",
                PHS_load="Pump Load per unit (MW)",
                PHS_max_hours="Pumped Storage - Max Storage (GWh)")
    csp_f = dict(CSP_max_hours='CSP Storage (hours)')
    g_f = dict(fom="Fixed O&M Cost (R/kW/yr)",
               p_nom='2022 Capacity (MW)',
               name='Power Station Name',
               carrier='Carrier',
               build_year='Future Commissioning Date',
               decomdate='Decommissioning Date',
               x='GPS Longitude',
               y='GPS Latitude',
               status='Status',
               heat_rate='Heat Rate (GJ/MWh)',
               fuel_price='Fuel Price (R/GJ)',
               vom='Variable O&M Cost (R/MWh)',
               max_ramp_up='Max Ramp Up (MW/min)',
               max_ramp_down='Max Ramp Down (MW/min)',
               min_stable='Min Stable Level (%)',
               unit_size='Unit size (MW)',
               units='Number units',
               maint_rate='Typical annual maintenance rate (%)',
               out_rate='Typical annual forced outage rate (%)')
    return g_f, ps_f, csp_f

def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n

def _add_missing_carriers_from_costs(n, costs, carriers):
    missing_carriers = pd.Index(carriers).difference(n.carriers.index)
    if missing_carriers.empty: return

    emissions_cols = costs.columns.to_series()\
                           .loc[lambda s: s.str.endswith('_emissions')].values
    suptechs = missing_carriers.str.split('-').str[0]
    emissions = costs.loc[suptechs, emissions_cols].fillna(0.)
    emissions.index = missing_carriers
    n.import_components_from_dataframe(emissions, 'Carrier')

def load_costs(tech_costs, cost_scenario, config, elec_config, config_years):
    """
    set all asset costs and other parameters
    """
    cost_data = pd.read_excel(tech_costs, sheet_name=cost_scenario,index_col=list(range(2))).sort_index()
    
    # Interpolate for years in config file but not in cost_data excel file
    config_years_array = np.array(config_years)
    missing_year = config_years_array[~np.isin(config_years_array,cost_data.columns)]
    if len(missing_year) > 0:
        for i in missing_year: 
            cost_data.insert(0,i,np.nan) # add columns of missing year to dataframe
        cost_data_tmp = cost_data[cost_data.columns.difference(['unit', 'source'])].sort_index(axis=1)
        cost_data_tmp = cost_data_tmp.interpolate(axis=1)
        cost_data = pd.concat([cost_data_tmp, cost_data[['unit','source']]],ignore_index=False,axis=1)

    # correct units to MW and ZAR
    cost_data.loc[cost_data.unit.str.contains("/kW"), config_years] *= 1e3
    cost_data.loc[cost_data.unit.str.contains("USD"), config_years] *= config["USD_to_ZAR"]
    cost_data.loc[cost_data.unit.str.contains("EUR"), config_years] *= config["EUR_to_ZAR"]

    costs = {}
    for y in config_years:
        costs[y]=cost_data.loc[idx[:, y]].unstack(level=1).fillna(
        {
            "CO2 intensity": 0,
            "FOM": 0,
            "VOM": 0,
            "discount rate": config["discountrate"],
            "efficiency": 1,
            "efficiency_store": 1,
            "efficiency_dispatch": 1,
            "fuel": 0,
            "investment": 0,
            "lifetime": 25,
        })

        costs[y]['efficiency_store']=costs[y]['efficiency'].pow(1./2) #if only 1 efficiency value is given assume it is round trip efficiency
        costs[y]['efficiency_dispatch']=costs[y]['efficiency'].pow(1./2)

        costs[y]["capital_cost"] = (
            (
                calculate_annuity(costs[y]["lifetime"], costs[y]["discount rate"])
                + costs[y]["FOM"] / 100.0
            )
            * costs[y]["investment"]
        )

        costs[y].at["OCGT", "fuel"] = costs[y].at["gas", "fuel"]
        costs[y].at["CCGT", "fuel"] = costs[y].at["gas", "fuel"]

        costs[y]["marginal_cost"] = costs[y]["VOM"] + costs[y]["fuel"] / costs[y]["efficiency"]

        costs[y] = costs[y].rename(columns={"CO2 intensity": "co2_emissions"})

        costs[y].at["OCGT", "co2_emissions"] = costs[y].at["gas", "co2_emissions"]
        costs[y].at["CCGT", "co2_emissions"] = costs[y].at["gas", "co2_emissions"]

        costs[y].at["solar", "capital_cost"] = 0.5 * (
            costs[y].at["solar-rooftop", "capital_cost"]
            + costs[y].at["solar-utility", "capital_cost"]
        )

        def costs_for_storage(store, link1, link2=None, max_hours=1.0):
            capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
            if link2 is not None:
                capital_cost += link2["capital_cost"]
            return pd.Series(
                dict(capital_cost=capital_cost, marginal_cost=0.0, co2_emissions=0.0)
            )

        max_hours = elec_config["max_hours"]
        costs[y].loc["battery"] = costs_for_storage(
            costs[y].loc["battery storage"],
            costs[y].loc["battery inverter"],
            max_hours=max_hours["battery"],
        )
        costs[y].loc['battery',:].fillna(costs[y].loc['battery inverter',:],inplace=True)
    
    for attr in ("marginal_cost", "capital_cost"):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs[y].loc[overwrites.index, attr] = overwrites

    return costs

def add_generator_availability(n,config_avail):
    # Add plant availability based on actual Eskom data
    eskom_data  = pd.read_excel(snakemake.input.existing_generators_eaf, sheet_name='eskom_data', na_values=['-'],index_col=[1,0],parse_dates=True)
    # if isinstance(n.snapshots, pd.MultiIndex):
    snapshots = n.snapshots.get_level_values(1)
    # else:
    #     snapshots=n.snapshots
    eaf_profiles = pd.DataFrame(1,index=snapshots,columns=[])
    delta_eaf=1 #TODO change with projections into the future

    # All existing generators in the Eskom fleet with available data
    for tech in n.generators.index[n.generators.index.isin(eskom_data.index.get_level_values(0).unique())]:
        reference_data = eskom_data.loc[tech].loc[eskom_data.loc[tech].index.year.isin(config_avail['reference_years'])]
        base_eaf=(reference_data['EAF %']/100).groupby(reference_data['EAF %'].index.month).mean()
        for y in n.investment_periods:
            eaf_profiles.loc[str(y),tech]=base_eaf[eaf_profiles.loc[str(y)].index.month].values * delta_eaf

    # New plants without existing data take best performing of Eskom fleet
    for carrier in ['coal', 'OCGT', 'CCGT', 'nuclear']:
        reference_data = eskom_data.loc[config_avail['new_unit_ref'][carrier]]
        reference_data = reference_data.loc[reference_data.index.year.isin(config_avail['years'])]
        base_eaf=(reference_data['EAF %']/100).groupby(reference_data['EAF %'].index.month).mean()* config_avail['new_unit_modifier'][carrier]
        for gen_ext in n.generators[(n.generators.carrier==carrier) & (n.generators.p_nom_extendable)].index:
            eaf_profiles[gen_ext]=1
            for y in n.investment_periods:
                eaf_profiles.loc[str(y),gen_ext]=base_eaf[eaf_profiles.loc[str(y)].index.month].values
     
    eaf_profiles[eaf_profiles>1]=1
    eaf_profiles.index=n.snapshots
    n.generators_t.p_max_pu[eaf_profiles.columns]=eaf_profiles

def add_min_stable_levels(n,generators,config_min_stable):
    # Existing generators
    for gen in generators.index: 
        if generators.loc[gen,'min_stable']!=np.nan:
            try:
                n.generators_t.p_min_pu[gen] = n.generators_t.p_max_pu[gen] * generators.loc[gen,'min_stable']   
                n.generators_t.p_max_pu[gen][n.generators_t.p_max_pu[gen]<generators.loc[gen,'min_stable']] = generators.loc[gen,'min_stable']   
            except:
                n.generators_t.p_min_pu[gen] = n.generators.p_max_pu[gen] * generators.loc[gen,'min_stable']   
                n.generators_t.p_max_pu[gen] = max(n.generators.p_max_pu[gen],generators.loc[gen,'min_stable'])
    
    # New plants without existing data take best performing of Eskom fleet
    for carrier in ['coal', 'OCGT', 'CCGT', 'nuclear']:
        for gen_ext in n.generators[(n.generators.carrier==carrier) & (~n.generators.index.isin(generators.index))].index:
            n.generators_t.p_min_pu[gen_ext] = n.generators_t.p_max_pu[gen_ext]*config_min_stable[carrier]
            n.generators_t.p_max_pu[gen_ext][n.generators_t.p_min_pu[gen_ext]<config_min_stable[carrier]] = config_min_stable[carrier]
    n.generators_t.p_min_pu=n.generators_t.p_min_pu.fillna(0)

 ## Attach components

# ### Load

def attach_load(n):
    load = pd.read_csv(snakemake.input.load)
    load = load.set_index(
        pd.to_datetime(load['SETTLEMENT_DATE'] + ' ' +
                       load['PERIOD'].astype(str) + ':00')
        .rename('t'))['SYSTEMENERGY']

    demand=pd.Series(0,index=n.snapshots)
    base_demand = (snakemake.config['electricity']['demand'] *
              normed(load.loc[str(snakemake.config['base_demand_year'])]))
    base_demand = remove_leap_day(base_demand)
    
    # if isinstance(n.snapshots, pd.MultiIndex):
    for y in n.investment_periods:
            demand.loc[y]=base_demand.values #TODO add annual growth in demand
    # else:
    #     demand = base_demand
    demand.index=n.snapshots
    
    n.madd("Load", n.buses.index,
           bus=n.buses.index,
           p_set=pdbcast(demand, normed(n.buses.population)))

### Set line costs

# def update_transmission_costs(n, costs):
#     opts = snakemake.config['lines']
#     for df in (n.lines, n.links):
#         if df.empty: continue

#         df['capital_cost'] = (df['length'] / opts['s_nom_factor'] *
#                               costs.at['Transmission lines', 'capital_cost'])

def update_transmission_costs(n, costs, length_factor=1.0, simple_hvdc_costs=False):
    for y in n.investment_periods:
        n.lines["capital_cost"] = (
            n.lines["length"] * length_factor * costs[y].at["HVAC overhead", "capital_cost"]
        )

        if n.links.empty:
            return

        dc_b = n.links.carrier == "DC"
        # If there are no "DC" links, then the 'underwater_fraction' column
        # may be missing. Therefore we have to return here.
        # TODO: Require fix
        if n.links.loc[n.links.carrier == "DC"].empty:
            return

        if simple_hvdc_costs:
            costs = (
                n.links.loc[dc_b, "length"]
                * length_factor
                * costs[y].at["HVDC overhead", "capital_cost"]
            )
        else:
            costs = (
                n.links.loc[dc_b, "length"]
                * length_factor
                * (
                    (1.0 - n.links.loc[dc_b, "underwater_fraction"])
                    * costs[y].at["HVDC overhead", "capital_cost"]
                    + n.links.loc[dc_b, "underwater_fraction"]
                    * costs[y].at["HVDC submarine", "capital_cost"]
                )
                + costs[y].at["HVDC inverter pair", "capital_cost"]
            )
        n.links.loc[dc_b, "capital_cost"] = costs


# ### Generators - TODO Update from pypa-eur
def attach_wind_and_solar(n, costs,wind_solar_profiles):
    g_f, ps_f, csp_f = map_generator_parameters() 
    input_files={'onwind_area': snakemake.input.onwind_area,
                 'solar_area': snakemake.input.solar_area}

    # Aggregate existing REIPPPP plants per region
    eskom_gens = pd.read_excel(snakemake.input.model_file, 
                                    sheet_name='existing_eskom_stations', 
                                    na_values=['-'],
                                    index_col=[0,1]).loc['base']
    eskom_gens = eskom_gens[eskom_gens['Carrier'].isin(['solar','onwind'])] # Currently only Sere wind farm
    ipp_gens = pd.read_excel(snakemake.input.model_file, 
                                    sheet_name='existing_non_eskom_stations', 
                                    na_values=['-'],
                                    index_col=[0,1]).loc['base']

    ipp_gens=ipp_gens[ipp_gens['Carrier'].isin(['solar','onwind'])] # add existing wind and PV IPP generators 
    gens = pd.concat([eskom_gens,ipp_gens])

    # Calculate fields where pypsa uses different conventions
    gens['marginal_cost'] = gens.pop(g_f['vom'])
    gens['capital_cost'] = 1e3*gens.pop(g_f['fom'])
    gens = gens.rename(columns={g_f[f]: f for f in {'p_nom', 'name', 'carrier', 'x', 'y','build_year','decomdate'}})
    gens['build_year'] = pd.to_datetime(gens['build_year'].fillna('{}-01-01'.format(snakemake.config['years'][0])).values).year 
    gens['decomdate'] = pd.to_datetime(gens['decomdate'].replace({'beyond 2050': '2051-01-01'}).values).year
    gens['lifetime'] = gens['decomdate'] - gens['build_year']
    gens = gens[gens.lifetime>0].drop(['decomdate','Status',
                                        g_f['maint_rate'],
                                        g_f['out_rate'],
                                        g_f['units'],
                                        g_f['unit_size'],
                                        g_f['min_stable']],axis=1)

    # Associate every generator with the bus of the region it is in or closest to
    pos = gpd.GeoSeries([Point(o.x, o.y) for o in gens[['x', 'y']].itertuples()], index=gens.index)
    regions = gpd.read_file(snakemake.input.supply_regions).set_index('name')
    for bus, region in regions.geometry.iteritems():
        pos_at_bus_b = pos.within(region)
        if pos_at_bus_b.any():
            gens.loc[pos_at_bus_b, "bus"] = bus
    gens.loc[gens.bus.isnull(), "bus"] = pos[gens.bus.isnull()].map(lambda p: regions.distance(p).idxmin())

    # Aggregate REIPPPP bid window generators at each bus #TODO use capacity weighted average for lifetime, costs 
    for carrier in ['solar','onwind']:
        plant_data = gens.loc[gens['carrier']==carrier,['Grouping','bus','p_nom']].groupby(['Grouping','bus']).sum()
        for param in ['lifetime','capital_cost','marginal_cost']:
            plant_data[param]=gens.loc[gens['carrier']==carrier,['Grouping','bus',param]].groupby(['Grouping','bus']).mean()

        for group in plant_data.index.levels[0]:
            # convert PPA price into an annualised cost -> i.e. take or pay. This forces the model to use the more expensive energy from REIPPPP
            capacity_factor = (wind_solar_profiles.loc[carrier][plant_data.loc[group].index]).mean()
            annual_cost = capacity_factor * 8760 * plant_data.loc[group,'marginal_cost']
            n.madd("Generator", plant_data.loc[group].index, suffix=" "+group+"_"+carrier,
                bus=plant_data.loc[group].index,
                carrier=carrier,
                build_year=n.investment_periods[0],
                lifetime=plant_data.loc[group,'lifetime'],
                p_nom_max = plant_data.loc[group,'p_nom'],
                p_nom_min = plant_data.loc[group,'p_nom'],
                p_nom_extendable=True,
                capital_cost=annual_cost,
                p_max_pu=wind_solar_profiles.loc[carrier][plant_data.loc[group].index].values,
                )

        # Add new generators
        active_area = pd.read_csv(input_files[carrier+'_area'], index_col=0).loc[lambda s: s.available_area > 0.]
        for y in n.investment_periods:
            n.madd("Generator", active_area.index, suffix=" "+carrier+"_"+str(y),
                bus=active_area.index,
                carrier=carrier,
                build_year=y,
                lifetime=costs[y].at[carrier,'lifetime'],
                p_nom_extendable=True,
                marginal_cost=costs[y].at[carrier, 'marginal_cost'],
                capital_cost=costs[y].at[carrier, 'capital_cost'],
                efficiency=costs[y].at[carrier, 'efficiency'],
                p_max_pu=wind_solar_profiles.loc[carrier][active_area.index].values)

# # Generators
def attach_existing_generators(n, costs, other_re_profiles):
    # Coal, gas, diesel, biomass, hydro, pumped storage
    g_f, ps_f, csp_f = map_generator_parameters()    

    # Add existing conventional generators that are active
    eskom_gens = pd.read_excel(snakemake.input.model_file, 
                                sheet_name='existing_eskom_stations',
                                na_values=['-'],
                                index_col=[0,1]).loc['base']
    eskom_gens = eskom_gens[~eskom_gens['Carrier'].isin(['solar','onwind'])]
    ipp_gens = pd.read_excel(snakemake.input.model_file,
                                sheet_name='existing_non_eskom_stations',
                                na_values=['-'],
                                index_col=[0,1]).loc['base']
    ipp_gens=ipp_gens[~ipp_gens['Carrier'].isin(['solar','onwind'])] # add existing non eskom generators (excluding solar, onwind)  
    gens = pd.concat([eskom_gens,ipp_gens])

    # Calculate fields where pypsa uses different conventions
    gens['efficiency'] = (3.6/gens.pop(g_f['heat_rate']))
    gens['marginal_cost'] = (3.6*gens.pop(g_f['fuel_price'])/gens['efficiency']).fillna(0) + gens.pop(g_f['vom'])
    gens['capital_cost'] = 1e3*gens.pop(g_f['fom'])
    gens['ramp_limit_up'] = 60*gens.pop(g_f['max_ramp_up'])/gens[g_f['p_nom']]
    gens['ramp_limit_down'] = 60*gens.pop(g_f['max_ramp_down'])/gens[g_f['p_nom']]

    gens = (gens.rename(columns={g_f[f]: f for f in {'p_nom', 'name', 'carrier', 'x', 'y','build_year','decomdate','min_stable'}})
            .rename(columns={ps_f[f]: f for f in {'PHS_efficiency','PHS_max_hours'}})
            .rename(columns={csp_f[f]: f for f in {'CSP_max_hours'}}))

    gens['build_year'] = pd.to_datetime(gens['build_year'].fillna('{}-01-01'.format(snakemake.config['years'][0])).values).year 
    gens['decomdate'] = pd.to_datetime(gens['decomdate'].replace({'beyond 2050': '2051-01-01'}).values).year
    gens['lifetime'] = gens['decomdate'] - gens['build_year']
    gens = gens[gens.lifetime>0].drop(['decomdate','Status',g_f['maint_rate'],g_f['out_rate'],g_f['units'],g_f['unit_size']],axis=1)

    # CahoraBassa will be added later, even though we don't have coordinates
    CahoraBassa  = pd.DataFrame(gens.loc["CahoraBassa"]).T
    # Drop power plants where we don't have coordinates or capacity
    gens = pd.DataFrame(gens.loc[lambda df: (df.p_nom>0.) & df.x.notnull() & df.y.notnull()])

    # Associate every generator with the bus of the region it is in or closest to
    pos = gpd.GeoSeries([Point(o.x, o.y) for o in gens[['x', 'y']].itertuples()], index=gens.index)
    regions = gpd.read_file(snakemake.input.supply_regions).set_index('name')
    for bus, region in regions.geometry.iteritems():
        pos_at_bus_b = pos.within(region)
        if pos_at_bus_b.any():
            gens.loc[pos_at_bus_b, "bus"] = bus
    gens.loc[gens.bus.isnull(), "bus"] = pos[gens.bus.isnull()].map(lambda p: regions.distance(p).idxmin())

    if snakemake.wildcards.regions=='RSA':
        CahoraBassa['bus'] = "RSA"
    elif snakemake.wildcards.regions=='9-supply':
        CahoraBassa['bus'] = "LIMPOPO"
    elif snakemake.wildcards.regions=='27-supply':
        CahoraBassa['bus'] = "POLOKWANE"
    gens = pd.concat([gens,CahoraBassa])

    gen_index=gens[gens.carrier.isin(['coal','nuclear','gas','diesel','hydro'])].index
    n.madd("Generator", gen_index,
        bus=gens.loc[gen_index,'bus'],
        carrier=gens.loc[gen_index,'carrier'],
        build_year=n.investment_periods[0],
        lifetime=gens.loc[gen_index,'lifetime'],
        p_nom = gens.loc[gen_index,'p_nom'],
        p_nom_extendable=False,
        ramp_limit_up = gens.loc[gen_index,'ramp_limit_up'],
        ramp_limit_down = gens.loc[gen_index,'ramp_limit_down'],
        marginal_cost=gens.loc[gen_index,'marginal_cost'],
        capital_cost=gens.loc[gen_index,'capital_cost'],
        #p_max_pu - added later under generator availability function
        )  
 
    for carrier in ['CSP','biomass']:
        plant_data = gens.loc[gens['carrier']==carrier,['Grouping','bus','p_nom']].groupby(['Grouping','bus']).sum()
        for param in ['lifetime','capital_cost','marginal_cost']:
            plant_data[param]=gens.loc[gens['carrier']==carrier,['Grouping','bus',param]].groupby(['Grouping','bus']).mean()

        for group in plant_data.index.levels[0]:
            # Duplicate Aggregate Eskom Data across the regions
            eskom_data = pd.concat([other_re_profiles.loc[carrier]] * (len(plant_data.loc[group].index)), axis=1, ignore_index=True)
            eskom_data.columns = plant_data.loc[group].index
            capacity_factor = (eskom_data[plant_data.loc[group].index]).mean()[0]
            annual_cost = capacity_factor * 8760 * plant_data.loc[group,'marginal_cost']

            n.madd("Generator", plant_data.loc[group].index, suffix=" "+group+"_"+carrier,
                bus=plant_data.loc[group].index,
                carrier=carrier,
                build_year=n.investment_periods[0],
                lifetime=plant_data.loc[group,'lifetime'],
                p_nom_max = plant_data.loc[group,'p_nom'],
                p_nom_min = plant_data.loc[group,'p_nom'],
                p_nom_extendable=True,
                capital_cost=annual_cost,
                p_max_pu=eskom_data.values,
                # purchase all power from existing IPPs despite higher marginal costs of early plants
                )    

    # ## HYDRO and PHS    
    # # Cohora Bassa imports to South Africa - based on Actual Eskom data from 2017-2022
    n.generators_t.p_max_pu['CahoraBassa'] = other_re_profiles.loc['imports'].values
    # Hydro power generation - based on actual Eskom data from 2017-2022
    for tech in n.generators[n.generators.carrier=='hydro'].index:
        n.generators_t.p_max_pu[tech] = other_re_profiles.loc['hydro'].values
    
    # PHS
    phs = gens[gens.carrier=='PHS']
    n.madd('StorageUnit', phs.index, carrier='PHS',
            bus=phs['bus'],
            p_nom=phs['p_nom'],
            max_hours=phs['PHS_max_hours'],
            capital_cost=phs['capital_cost'],
            marginal_cost=phs['marginal_cost'],
            p_max_pu=1.,  # dispatch
            p_min_pu=0.,  # store
            efficiency_dispatch=phs['PHS_efficiency']**(0.5),
            efficiency_store=phs['PHS_efficiency']**(0.5),
            cyclic_state_of_charge=True
            #inflow=inflow_t.loc[:, hydro.index]) #TODO add in
            )

    _add_missing_carriers_from_costs(n, costs[snakemake.config['years'][0]], gens.carrier.unique())

    return gens

def attach_extendable_generators(n, costs, other_re_profiles):
    elec_opts = snakemake.config['electricity']
    carriers = elec_opts['extendable_carriers']['Generator']
    buses = elec_opts['buses'][snakemake.wildcards.regions]

    _add_missing_carriers_from_costs(n, costs[snakemake.config['years'][0]], carriers)

    for y in snakemake.config['years']: 
        for carrier in carriers:
            buses_i = buses.get(carrier, n.buses.index)
            n.madd("Generator", buses_i, suffix=" " + carrier + "_"+str(y),
                bus=buses_i,
                p_nom_extendable=True,
                carrier=carrier,
                build_year=y,
                lifetime=costs[y].at[carrier, 'lifetime'],
                capital_cost=costs[y].at[carrier, 'capital_cost'],
                marginal_cost=costs[y].at[carrier, 'marginal_cost'],
                efficiency=costs[y].at[carrier, 'efficiency'])

def attach_storage(n, costs):
    elec_opts = snakemake.config['electricity']
    carriers = elec_opts['extendable_carriers']['StorageUnit']
    max_hours = elec_opts['max_hours']
    buses = elec_opts['buses']

    _add_missing_carriers_from_costs(n, costs[snakemake.config['years'][0]], carriers)

    for y in snakemake.config['years']:
        for carrier in carriers:
            buses_i = buses.get(carrier, n.buses.index)
            n.madd("StorageUnit", buses_i, " " + carrier + "_" + str(y),
                bus=buses_i,
                p_nom_extendable=True,
                carrier=carrier,
                build_year=y,
                capital_cost=costs[y].at[carrier, 'capital_cost'],
                marginal_cost=costs[y].at[carrier, 'marginal_cost'],
                efficiency_store=costs[y].at[carrier, 'efficiency_store'],
                efficiency_dispatch=costs[y].at[carrier, 'efficiency_dispatch'],
                max_hours=max_hours[carrier],
                cyclic_state_of_charge=True)

def add_co2limit(n):
    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=snakemake.config['electricity']['co2limit'])

def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)

def add_peak_demand_hour_without_variable_feedin(n):
    new_hour = n.snapshots[-1] + pd.Timedelta(hours=1)
    n.set_snapshots(n.snapshots.append(pd.Index([new_hour])))

    # Don't value new hour for energy totals
    n.snapshot_weightings[new_hour] = 0.

    # Don't allow variable feed-in in this hour
    n.generators_t.p_max_pu.loc[new_hour] = 0.

    n.loads_t.p_set.loc[new_hour] = (
        n.loads_t.p_set.loc[n.loads_t.p_set.sum(axis=1).idxmax()]
        * (1.+snakemake.config['electricity']['SAFE_reservemargin'])
    )

def add_nice_carrier_names(n, config):
    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = nice_names
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(
            f"tech_colors for carriers {missing_i} not defined " "in config."
        )
    n.carriers["color"] = colors


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('add_electricity', **{'costs':'za_original',
                            'regions':'27-supply',#'27-supply',
                            'resarea':'redz',
                            'll':'copt',
                            'opts':'LC',#-30SEG',
                            'attr':'p_nom'})

    opts = snakemake.wildcards.opts.split('-')
    n = pypsa.Network(snakemake.input.base_network)
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.wildcards.costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        snakemake.config["years"],
    )

    wind_solar_profiles = xr.open_dataset(snakemake.input.wind_solar_profiles).to_dataframe()
    other_re_profiles = xr.open_dataset(snakemake.input.other_re_profiles).to_dataframe()
    
    attach_load(n)
    if snakemake.wildcards.regions!='RSA':
        update_transmission_costs(n, costs)
    gens = attach_existing_generators(n, costs, other_re_profiles)
    attach_wind_and_solar(n, costs, wind_solar_profiles)
    attach_extendable_generators(n, costs, wind_solar_profiles)
    attach_storage(n, costs)
    #if snakemake.config['electricity']['generator_availability']['implement_availability']==True:
    #    add_generator_availability(n,snakemake.config['electricity']['generator_availability'])
    #    add_min_stable_levels(n,gens,snakemake.config['electricity']['min_stable_levels'])       
    add_nice_carrier_names(n, snakemake.config)
    n.export_to_netcdf(snakemake.output[0])
