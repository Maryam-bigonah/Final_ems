import pandas as pd
df = pd.read_csv('results/scenario1/scenario1_dispatch_timeseries.csv')
mask = (df['grid_import'] <= 1e-6) & (df['batt_charge_grid'] > 1e-6)
print(f'Count: {mask.sum()}')
if mask.sum() > 0:
    print(f'Max batt_charge_grid when grid_import==0: {df.loc[mask, "batt_charge_grid"].max()}')
    print(f'Total batt_charge_grid energy when grid_import==0: {df.loc[mask, "batt_charge_grid"].sum()}')
