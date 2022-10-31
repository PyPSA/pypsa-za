import pypsa
import geopandas as gpd
import numpy as np
import pandas as pd
import time
import xarray as xr
from _helpers import configure_logging

#logger = logging.getLogger(__name__)

def remove_leap_day(df):
    return df[~((df.index.month == 2) & (df.index.day == 29))]

if __name__ == '__main__':
    renewable_carriers=['solar','onwind','CSP','hydro','biomass']
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_renewable_profiles', 
                            **{'costs':'ambitions',
                            'regions':'27-supply',#'27-supply',
                            'resarea':'redz',
                            'll':'copt',
                            'opts':'LC',#-30SEG',
                            'attr':'p_nom'})

    n = pypsa.Network(snakemake.input.base_network)

    carrier_iter=0
    for carrier in ['solar','onwind','CSP','biomass','hydro','imports']:
        n.add("Carrier", name=carrier)
        weather_years=snakemake.config['years']['reference_weather_years'][carrier]
        for i in range(0,int(np.ceil(len(n.investment_periods)/len(weather_years))-1)):
            weather_years+=weather_years
   
        pu_data = (pd.read_excel(snakemake.input.profiles,skiprows=[1], 
                                    sheet_name=carrier,index_col=0,parse_dates=True)
                                    .resample('1h').mean())
        pu_data  = remove_leap_day(pu_data)
        cnt=0
        if carrier in ['solar','onwind']:
            resource_carrier=pd.DataFrame(0,index=n.snapshots.levels[1],columns=n.buses.index)
        else:
            resource_carrier=pd.DataFrame(0,index=n.snapshots.levels[1],columns=['ESKOM DATA'])
        # if no data exists for a bus region, use the default RSA hourly data (from Eskom)
        pu_data=pu_data[resource_carrier.columns]
        for y in n.investment_periods:    
            resource_carrier.loc[str(y)] = (pu_data.loc[str(weather_years[cnt])]
                                        .clip(lower=0., upper=1.)).values     
            cnt+=1
        resource_carrier['carrier']=carrier
        
        if carrier_iter == 0:
            resource = resource_carrier
            carrier_iter=1
        else:
            resource= pd.concat([resource,resource_carrier])
        
        if carrier=='onwind':
            resource.index = pd.MultiIndex.from_arrays([resource.carrier,resource.index])
            resource.drop('carrier',axis=1,inplace=True)
            resource.to_xarray().to_netcdf(snakemake.output.wind_solar_profiles)
            carrier_iter=0
    
        if carrier=='imports':
            resource.index = pd.MultiIndex.from_arrays([resource.carrier,resource.index])
            resource.drop('carrier',axis=1,inplace=True)
            resource.to_xarray().to_netcdf(snakemake.output.other_re_profiles)
            carrier_iter=0



    