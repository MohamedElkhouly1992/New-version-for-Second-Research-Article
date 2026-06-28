from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)
rows=[]
strategies=['S0','S1','S2','S3']
climates=['hot_arid']
for strategy in strategies:
    for year in range(1,6):
        for severity in np.linspace(0.05,0.75,8):
            for m in range(12):
                control = {'S0':0.00,'S1':0.08,'S2':0.14,'S3':0.22}[strategy]
                delta = np.clip(severity + 0.018*year - control*severity + np.random.normal(0,0.015),0,1)
                energy = 950 + 420*delta + 12*m + 25*year - 160*control + np.random.normal(0,25)
                comfort = 0.45 + 1.8*delta - 0.35*control + np.random.normal(0,0.08)
                cop = 4.2 - 1.1*delta + 0.15*control + np.random.normal(0,0.05)
                cost = energy*95
                co2 = energy*0.42
                rows.append(dict(strategy=strategy,severity=severity,climate='hot_arid',year=year,month=m+1,
                                 annual_energy_MWh=energy,annual_cost_usd=cost,annual_co2_tonne=co2,
                                 mean_COP=cop,mean_delta=delta,mean_comfort_dev=comfort,
                                 occupied_discomfort_days=max(0, int(comfort*18+np.random.normal(0,2)))))
df=pd.DataFrame(rows)
out=Path('demo_hvac_5year_severity_strategy.csv')
df.to_csv(out,index=False)
print(out.resolve())
