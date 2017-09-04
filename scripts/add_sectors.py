# coding: utf-8

import pypsa
import pandas as pd
import numpy as np
import os
from six import iteritems

from vresutils.costdata import annuity, USD2013_to_EUR2013

from _helpers import madd, pdbcast

def normed(s): return s/s.sum()

if 'snakemake' not in globals():
    from vresutils import Dict
    import yaml
    snakemake = Dict()
    snakemake.input = Dict(network='../networks/elec_CSIR-Expected-Apr2016_redz_Co2L',
                           emobility='../data/external/emobility/')
    snakemake.wildcards = Dict(sectors="E+BEV", mask="redz", opts="Co2L", cost="CSIR-Expected-Apr2016")
    snakemake.output = ['../networks/sector_CSIR-Expected-Apr2016_redz_E+BEV_Co2L']
    with open('../config.yaml') as f:
        snakemake.config = yaml.load(f)

###########################################################################################

def generate_periodic_profiles(dt_index, freq="H", weekly_profile=range(24*7)):

    #Give a 24*7 long list of weekly hourly profiles
    weekly_profile = pd.Series(weekly_profile, range(24*7))
    hour_of_the_week = pd.Series(24*dt_index.weekday+dt_index.hour, dt_index)

    return hour_of_the_week.map(weekly_profile)


def add_transport(n, V2G=True):
    buses = n.buses.index[n.buses.population > 0.]
    population = n.buses.loc[buses, 'population']

    opts = snakemake.config['transport']

    emobility = snakemake.input.emobility
    weekly_profile_kfz = pd.read_csv(os.path.join(emobility, "KFZ__count"), skiprows=2)["count"]
    weekly_profile_pkw = pd.read_csv(os.path.join(emobility, "Pkw__count"), skiprows=2)["count"]

    transport_demand = (
        normed(generate_periodic_profiles(n.snapshots, weekly_profile=weekly_profile_kfz.values))
        * (opts['energy_total'] / opts.get('efficiency_gain', 1.))
    )

    def renormalize(s, vmax, vmean):
        return vmax - (vmax - vmean) * (s - s.min())/(s.mean() - s.min())
    battery_availability = generate_periodic_profiles(
        n.snapshots,
        weekly_profile=renormalize(
            weekly_profile_pkw,
            vmax=opts['availability_max'],
            vmean=opts['availability_mean']
        ))


    n.add("Carrier", "Li ion")
    buses_ev_battery = madd(n, "Bus", name="EV battery", index=buses, carrier="Li ion")
    madd(n, "Load", index=buses,
         bus=buses_ev_battery,
         p_set=pdbcast(transport_demand, normed(population)))

    cars = normed(population) * opts['total_cars']
    charging_discharging_power = cars * opts['car_battery_p_nom']

    madd(n, "Store", name="battery storage", index=buses,
         bus=buses_ev_battery,
         e_cyclic=True,
         e_nom=cars * opts['car_battery_e_nom'],
         standing_loss=opts['standing_loss'])

    madd(n, "Link", name="BEV charger", index=buses,
         bus0=buses, bus1=buses_ev_battery,
         p_nom=charging_discharging_power,
         efficiency=opts['efficiency'],
         p_max_pu=battery_availability,
         #These were set non-zero to find LU infeasibility when availability = 0.25
         #p_nom_extendable=True,
         #p_nom_min=p_nom,
         #capital_cost=1e6,  #i.e. so high it only gets built where necessary
    )

    if V2G:
        madd(n, "Link", name="V2G", index=buses,
             bus0=buses_ev_battery, bus1=buses,
             p_nom=charging_discharging_power,
             p_max_pu=battery_availability,
             efficiency=opts['efficiency'])

    #TO DO
    # network.add("Load",node + " transport fuel cell",
    #             bus=node + " H2",
    #             p_set=options['transport_fuel_cell_share']/0.58*transport[node].values[:8760],
    #            )


def add_gas_infrastructure(n, costs):
    buses = n.buses.index[n.buses.population > 0.]
    discountrate = snakemake.config['costs']['discountrate']

    n.add("Carrier", "H2")
    buses_h2 = madd(n, "Bus", name="H2", index=buses, carrier="H2")
    madd(n, "Link", name="H2 Electrolysis", index=buses,
         bus0=buses, bus1=buses_h2,
         p_nom_extendable=True,
         efficiency=0.75,
         #Cost from http://www.nrel.gov/docs/fy09osti/45873.pdf "Future Timeframe"
         #(same source as Budishak)
         capital_cost=(annuity(20.,discountrate)+0.017)*300.*1000.*USD2013_to_EUR2013)
    madd(n, "Link", name="H2 Fuel Cell", index=buses,
         bus0=buses_h2, bus1=buses,
         p_nom_extendable=True,
         efficiency=0.58,
         #Cost from http://www.nrel.gov/docs/fy09osti/45873.pdf "Future Timeframe"
         #(same source as Budishak)
         #NB: Costs refer to electrical side, so must multiply by efficiency
         capital_cost=(annuity(20.,discountrate)+0.017)*437.*0.58*1000.*USD2013_to_EUR2013)
    madd(n, "Store", name="Store", bus=buses_h2,
         e_nom_extendable=True,
         e_cyclic=True,
         capital_cost=annuity(20.,discountrate)*11.2*1000.*USD2013_to_EUR2013)


    # TODO
    #OCGT bus w/gas generation link
    # network.add("Bus",
    #             node + " OCGT",
    #             carrier="OCGT")
    # network.add("Link",
    #             node + " OCGT",
    #             bus0=node+ " OCGT",
    #             bus1=node,
    #             capital_cost=cc['efi']*cc['wki']*Nyears,
    #             p_nom_extendable=True,
    #             efficiency=cc['efi'])

###############
#### heat

def compute_heat_demand(n):
    # shares ={}
    # file_name = snakemake.input.heating_residential
    # shares["Residential"] = (pd.read_csv(file_name,index_col=0).T["DE"]).T
    # file_name = snakemake.input.heating_tertiary
    # shares["Services"] = (pd.read_csv(file_name,index_col=0).T["DE"]).T

    # sectors = ["Residential","Services"]

    # for sector in sectors:
    #     energy_totals[sector +" Combustion"] = energy_totals[sector] - energy_totals[sector + " Electricity"]
    #     energy_totals[sector +" Space"] = shares[sector].space*energy_totals[sector + " Combustion"]
    #     energy_totals[sector +" Water"] = shares[sector].water*energy_totals[sector + " Combustion"]

    # energy_totals["Space Heating"] = energy_totals[[sector + " Space" for sector in sectors]].sum()
    # energy_totals["Water Heating"] = energy_totals[[sector + " Water" for sector in sectors]].sum()

    heating_space = 0.
    heating_water = 0. # TODO

    heat_demand = heating_water / 8760

    if heating_space > 0.:
        daily_heat_demand = pd.read_csv(snakemake.input.heat_demand, index_col=0, parse_dates=True)
        intraday_profiles = pd.read_csv(snakemake.input.heat_profile, index_col=0)
        heat_demand_space = normed(daily_heat_demand.reindex(n.snapshots, method='ffill').multiply(
            generate_periodic_profiles(
                n.snapshots,
                weekly_profile=np.r_[np.repeat(intraday_profiles['weekday'], 5), np.repeat(intraday_profiles['weekend'], 2)]),
            axis=0)) * heating_space

        # cop = (pd.read_csv(snakemake.input.cop, index_col=0, parse_dates=True))

        heat_demand = heat_demand + heat_demand_space

    return heat_demand

def add_water_heating(n):
    ##### CHP Parameters
    ###### electrical efficiency with no heat output
    eta_elec = 0.468
    ###### loss of fuel for each addition of heat
    c_v = 0.15
    ###### backpressure ratio
    c_m = 0.75
    ###### ratio of max heat output to max electrical output
    p_nom_ratio = 1.


    heat_demand = compute_heat_demand(n)

    network.add("Carrier", "heat")

    network.add("Bus", node + " heat", carrier="heat")
    network.add("Link", node + " heat pump",
                bus0=node,
                bus1=node + " heat",
                efficiency=cop[node], #cop for 2011 time_dep_hp_cop
                capital_cost=(annuity(20,discountrate)+0.015)*3.*1.050e6, #20a, 1% FOM, 1050 EUR/kWth from [HP] NB: PyPSA uses bus0 for p_nom restriction, hence factor 3 to get 3150 EUR/kWel
                p_nom_extendable=True
               )
    network.add("Load", node + " heat",
                bus=node + " heat",
                p_set= heat_demand[node]
               )
    network.add("Link", node + " resistive heater",
                bus0=node,
                bus1=node + " heat",
                efficiency=0.9,
                capital_cost=(annuity(20,discountrate)+0.02)*0.9*1.e5, #100 EUR/kW_th, 2% FOM from Schaber thesis
                p_nom_extendable=True,
               )


    ##### H2 bus w/methanation
    # methanation
    network.add("Link",
                node + " Sabatier",
                bus0=node+" H2",
                bus1=node+" OCGT",
                p_nom_extendable=True,
                #Efficiency from Katrin Schaber PhD thesis
                efficiency=0.77,
                #Costs from Katrin Schaber PhD thesis; comparable with HP (1.5e6 including H2 electrolysis)
                capital_cost=(annuity(20.,discountrate)+0.02)*1.1e6*Nyears)
    # gas boiler
    network.add("Link", node + " gas boiler",
                p_nom_extendable=True,
                bus0=node + " OCGT",
                bus1=node + " heat",
                capital_cost=(annuity(20,0.07)+0.01)*0.9*3.e5, #300 EUR/kW_th, 1% FOM from Schaber thesis, 20a from HP
                efficiency=0.9,)
    # chp - standardmäßig rein? Ist es jetzt auf jeden Fall!
    network.add("Link", node + " CHP electric",
                bus0=node+ " OCGT",
                bus1=node,
                capital_cost=(annuity(25,0.07)+0.03)*1.4e6*eta_elec, #From HP decentral
                efficiency=eta_elec,
                p_nom_extendable=True)
    network.add("Link", node + " CHP heat",
                p_nom_extendable=True,
                bus0=node + " OCGT",
                bus1=node + " heat",
                capital_cost=0.,
                efficiency=eta_elec/c_v)



    ##### heat flexibilities:
    if "TES" in flexibilities:
        network.add("Carrier","water tanks")
        network.add("Bus", node + " water tanks", carrier="water tanks")
        network.add("Link", node + " water tanks charger",
                    bus0=node + " heat",
                    bus1=node + " water tanks",
                    efficiency=0.9,
                    capital_cost=0.,
                    p_nom_extendable=True
                   )
        network.add("Link", node + " water tanks discharger",
                    bus0=node + " water tanks",
                    bus1=node + " heat",
                    efficiency=0.9,
                    capital_cost=0.,
                    p_nom_extendable=True
                   )
        network.add("Store", node + " water tank",
                    bus=node + " water tanks",
                    e_cyclic=True,
                    e_nom_extendable=True,
                    standing_loss=1-np.exp(-1/(24.*180)), #options["tes_tau"])),  # [HP] 180 day time constant for centralised, 3 day for decentralised
                    capital_cost=(annuity(40,discountrate)+0.01)*20/(1.17e-3*40), #[HP] 40 years, 20 EUR/m^3 in EUR/MWh for 40 K diff and 1.17 kWh/m^2, 1% FOM
                   )
####################################################################






#################################################################################################################


if __name__ == "__main__":
    n = pypsa.Network(snakemake.input.network)
    sectors = set(snakemake.wildcards.sectors.split('+'))

    if 'BEV' in sectors:
        add_transport(n, V2G='V2G' in sectors)

    # if 'WEH' in sectors:
    #     add_water_heating(n)

    n.export_to_csv_folder(snakemake.output[0])

