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

    # correct units to MW and EUR
    cost_data.loc[cost_data.unit.str.contains("/kW"), config_years] *= 1e3
    cost_data.loc[cost_data.unit.str.contains("USD"), config_years] *= config["USD_to_EUR"]
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

# ## Attach components

# ### Load

def attach_load(n):
    load = pd.read_csv(snakemake.input.load)
    load = load.set_index(
        pd.to_datetime(load['SETTLEMENT_DATE'] + ' ' +
                       load['PERIOD'].astype(str) + ':00')
        .rename('t')
    )['SYSTEMENERGY']

    demand=pd.Series(0,index=n.snapshots)
    base_demand = (snakemake.config['electricity']['demand'] *
              normed(load.loc[str(snakemake.config['base_demand_year'])]))
    base_demand = remove_leap_day(base_demand)
    
    if len(n.investment_periods)==1:
        demand = base_demand.values
    else:
        for y in n.investment_periods:
                demand.loc[y]=base_demand.values #TODO add annual growth in demand
    
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
def attach_wind_and_solar(n, costs):
    capacity_per_sqm = snakemake.config['respotentials']['capacity_per_sqm']
    # repeat weather years to fill multi horizon
    len_years = len(n.investment_periods)
    weather_years=snakemake.config['base_weather_years']
    len_weather_years = len(weather_years)
    for i in range(0,int(np.ceil(len_years/len_weather_years)-1)):
        weather_years+=weather_years

    ## Onshore wind
    n.add("Carrier", name="onwind")
    onwind_area = pd.read_csv(snakemake.input.onwind_area, index_col=0).loc[lambda s: s.available_area > 0.]
    onwind_res=pd.DataFrame(0,index=n.snapshots,columns=onwind_area.index)
    onwind_data = (pd.read_excel(snakemake.input.onwind_profiles,
                                skiprows=[1], sheet_name='Wind power profiles')
                                .rename(columns={'supply area\'s name': 't'}).set_index('t')
                                .resample('1h').mean())
    onwind_data = remove_leap_day(onwind_data)

    cnt=0
    #
    if len_years==1:
        onwind_res = (onwind_data.loc[str(weather_years[cnt])]
                            .reindex(columns=onwind_area.index)
                            .clip(lower=0., upper=1.)).values     
    else:
        for y in n.investment_periods:    

            onwind_res.loc[y] = (onwind_data.loc[str(weather_years[cnt])]
                                .reindex(columns=onwind_area.index)
                                .clip(lower=0., upper=1.)).values     
            cnt+=1
        
    for y in n.investment_periods:
        n.madd("Generator", onwind_area.index, suffix=" onwind_"+str(y),
            bus=onwind_area.index,
            carrier="onwind",
            build_year=y,
            lifetime=20,
            p_nom_extendable=True,
            #p_nom_max=onwind_area.available_area * capacity_per_sqm['onwind'],
            marginal_cost=costs[y].at['onwind', 'marginal_cost'],
            capital_cost=costs[y].at['onwind', 'capital_cost'],
            efficiency=costs[y].at['onwind', 'efficiency'],
            p_max_pu=onwind_res)

    ## Solar PV
    n.add("Carrier", name="solar")
    solar_area = pd.read_csv(snakemake.input.solar_area, index_col=0).loc[lambda s: s.available_area > 0.]
    solar_res=pd.DataFrame(0,index=n.snapshots,columns=solar_area.index)
    solar_data = (pd.read_excel(snakemake.input.solar_profiles,
                                skiprows=[1], sheet_name='PV profiles')
                                .rename(columns={'supply area\'s name': 't'}).set_index('t')
                                .resample('1h').mean())
    solar_data = remove_leap_day(solar_data)

    cnt=0
    if len_years==1:
        solar_res = (solar_data.loc[str(weather_years[cnt])]
                            .reindex(columns=solar_area.index)
                            .clip(lower=0., upper=1.)).values     
    else:
        for y in n.investment_periods:    

            solar_res.loc[y] = (solar_data.loc[str(weather_years[cnt])]
                                .reindex(columns=solar_area.index)
                                .clip(lower=0., upper=1.)).values     
            cnt+=1

    for y in n.investment_periods:
        n.madd("Generator", solar_area.index, suffix=" solar_"+str(y),
            bus=solar_area.index,
            carrier="solar",
            build_year=y,
            lifetime=25,
            p_nom_extendable=True,
            #p_nom_max=solar_area.available_area * capacity_per_sqm['solar'],
            marginal_cost=costs[y].at['solar', 'marginal_cost'],
            capital_cost=costs[y].at['solar', 'capital_cost'],
            efficiency=costs[y].at['solar', 'efficiency'],
            p_max_pu=solar_res)

# # Generators
def attach_existing_generators(n, costs):
    historical_year = snakemake.config['historical_year']
    len_years = len(n.investment_periods)
    weather_years=snakemake.config['base_weather_years']
    len_weather_years = len(weather_years)
    for i in range(0,int(np.ceil(len_years/len_weather_years)-1)):
        weather_years+=weather_years
        
    ps_f = dict(efficiency="Pump Efficiency (%)",
                pump_units="Pump Units",
                pump_load="Pump Load per unit (MW)",
                max_storage="Pumped Storage - Max Storage (GWh)")

    csp_f = dict(max_hours='CSP Storage (hours)')

    g_f = dict(fom="Fixed Operations and maintenance costs (R/kW/yr)",
               p_nom='Installed/ Operational Capacity in 2016 (MW)',
               name='Power Station Name',
               carrier='Fuel/technology type',
               build_year='Commissioning Date',
               decomdate='Decommissioning Date',
               x='GPS Longitude',
               y='GPS Latitude',
               status='Status',
               heat_rate='Heat Rate (GJ/MWh)',
               fuel_price='Fuel Price (R/GJ)',
               vom='Variable Operations and Maintenance Cost (R/MWh)',
               max_ramp_up='Max Ramp Up (MW/min)',
               unit_size='Unit size (MW)',
               units='Number units',
               maint_rate='Typical annual maintenance rate (%)',
               out_rate='Typical annual forced outage rate (%)',
               owner='Owner')

    gens = pd.read_excel(snakemake.input.existing_generators, na_values=['-'])
    
    # Make field "Fixed Operations and maintenance costs" numeric
    includescapex_i = gens[g_f['fom']].str.endswith(' (includes capex)').dropna().index
    gens.loc[includescapex_i, g_f['fom']] = gens.loc[includescapex_i, g_f['fom']].str[:-len(' (includes capex)')]
    gens[g_f['fom']] = pd.to_numeric(gens[g_f['fom']])


    # Calculate fields where pypsa uses different conventions
    gens['efficiency'] = 3.6/gens.pop(g_f['heat_rate'])
    gens['marginal_cost'] = 3.6*gens.pop(g_f['fuel_price'])/gens['efficiency'] + gens.pop(g_f['vom'])
    gens['capital_cost'] = 1e3*gens.pop(g_f['fom'])
    gens['ramp_limit_up'] = 60*gens.pop(g_f['max_ramp_up'])/gens[g_f['p_nom']]
    
    gens = gens.rename(columns={g_f[f]: f for f in {'p_nom', 'name', 'carrier', 'x', 'y','build_year','decomdate'}})
    gens['build_year'] = pd.to_datetime(gens['build_year'].fillna('{}-01-01'.format(n.investment_periods[0])).values).year 
    gens['decomdate'] = pd.to_datetime(gens['decomdate'].replace({'beyond 2050': '2051-01-01'}).values).year
    gens['lifetime'] = gens['decomdate'] - gens['build_year']
    gens = gens[gens.lifetime>0].drop(['decomdate','Status','Owner',g_f['maint_rate'],g_f['out_rate'],g_f['units'],g_f['unit_size']],axis=1)
    gens.set_index('name',inplace=True)

    # CahoraBassa will be added later, even though we don't have coordinates
    CahoraBassa = gens.loc["CahoraBassa"]

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
    elif snakemake.wildcards.regions=='27-supply':
        CahoraBassa['bus'] = "POLOKWANE"
    gens = gens.append(CahoraBassa)

    # Now we split them by carrier and have some more carrier specific cleaning
    gens.carrier.replace({"Pumped Storage": "PHS"}, inplace=True)

    # HYDRO - currently only a single year of data

    n.add("Carrier", "hydro")
    n.add("Carrier", "PHS")

    hydro = pd.DataFrame(gens.loc[gens.carrier.isin({'PHS', 'hydro'})])
    hydro["efficiency_store"] = hydro["efficiency_dispatch"] = np.sqrt(hydro.pop(ps_f['efficiency'])/100.).fillna(1.)

    hydro["max_hours"] = 1e3*hydro.pop(ps_f["max_storage"])/hydro["p_nom"]

    hydro["p_min_pu"] = - (hydro.pop(ps_f["pump_load"]) * hydro.pop(ps_f["pump_units"]) / hydro["p_nom"]).fillna(0.)

    hydro = (hydro
             .assign(p_max_pu=1.0, cyclic_state_of_charge=True)
             .drop(list(csp_f.values()) + ['ramp_limit_up', 'efficiency'], axis=1))

    hydro.max_hours.fillna(hydro.max_hours.mean(), inplace=True)

#TODO fix this for multi-horizon
    # hydro_inflow_data = pd.read_csv(snakemake.input.hydro_inflow, index_col=0, parse_dates=True)
    # hydro_inflow_data = remove_leap_day(hydro_inflow_data)
    # hydro_res = pd.DataFrame(0,index=n.snapshots,columns=hydro.index)

    # if len(n.investment_periods)==1:
    #     hydro_inflow = hydro_inflow_data.loc[str(weather_years[0])]
    #     hydro_inflow_za = pd.DataFrame(hydro_inflow[['ZA']].values * normed(hydro.loc[hydro_za_b, 'p_nom'].values),
    #                                 columns=hydro.index[hydro_za_b], index=hydro_inflow.index)
    #     hydro_inflow_za['CahoraBassa'] = hydro.at['CahoraBassa', 'p_nom']/2187.*hydro_inflow['MZ']
    # else:
    #     for y in n.investment_periods:    

    #         hydro_res.loc[y] = (solar_data.loc[str(weather_years[cnt])]
    #                             .reindex(columns=solar_area.index)
    #                             .clip(lower=0., upper=1.)).values     
    #         cnt+=1
    hydro_inflow_za=pd.DataFrame(0.1,index=n.snapshots,columns=hydro.index)
    hydro.marginal_cost.fillna(0., inplace=True)
    n.import_components_from_dataframe(hydro, "StorageUnit")
    n.import_series_from_dataframe(hydro_inflow_za, "StorageUnit", "inflow")

    if snakemake.config['electricity'].get('csp'):
        n.add("Carrier", "CSP")
        csp = (pd.DataFrame(gens.loc[gens.carrier == "CSP"])
               .drop(list(ps_f.values()) + ["ramp_limit_up", "efficiency"], axis=1)
               .rename(columns={csp_f['max_hours']: 'max_hours'}))

        # TODO add to network with time-series and everything
    gens = (gens.loc[gens.carrier.isin({"coal", "nuclear"})]
            .drop(list(ps_f.values()) + list(csp_f.values()), axis=1))
    _add_missing_carriers_from_costs(n, costs[n.investment_periods[0]], gens.carrier.unique())

    n.import_components_from_dataframe(gens, "Generator")

def attach_extendable_generators(n, costs):
    elec_opts = snakemake.config['electricity']
    carriers = elec_opts['extendable_carriers']['Generator']
    buses = elec_opts['buses'][snakemake.wildcards.regions]

    _add_missing_carriers_from_costs(n, costs[n.investment_periods[0]], carriers)

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

    _add_missing_carriers_from_costs(n, costs[n.investment_periods[0]], carriers)

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
        snakemake = mock_snakemake('add_electricity', **{'costs':'ambitions',
                            'regions':'27-supply',
                            'resarea':'redz',
                            'll':'copt',
                            'opts':'LC-30SEG',
                            'attr':'p_nom'})

    opts = snakemake.wildcards.opts.split('-')
    n = pypsa.Network(snakemake.input.base_network)
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.wildcards.costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        snakemake.config["years"],
    )

    attach_load(n)
    if snakemake.wildcards.regions!='RSA':
        update_transmission_costs(n, costs)
    attach_existing_generators(n, costs)
    attach_wind_and_solar(n, costs)
    attach_extendable_generators(n, costs)
    attach_storage(n, costs)
    add_nice_carrier_names(n, snakemake.config)

    n.export_to_netcdf(snakemake.output[0])
