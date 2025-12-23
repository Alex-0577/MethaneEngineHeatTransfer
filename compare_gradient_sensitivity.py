import os
import pandas as pd

dirs = [
    'AE-1310_grad_20000',
    'AE-1310_grad_10000',
    'AE-1310_grad_5000',
    'AE-1310_grad_1000'
]
rows = []
for d in dirs:
    base = os.path.join(d, 'CSV_Data')
    td = os.path.join(base, 'temperature_distribution.csv')
    cf = os.path.join(base, 'coolant_flow_parameters.csv')
    ht = os.path.join(base, 'heat_transfer_parameters.csv')
    if not os.path.exists(td):
        print('missing', td)
        continue
    tdf = pd.read_csv(td)
    cff = pd.read_csv(cf)
    htf = pd.read_csv(ht)
    max_wt = tdf['gas_side_wall_temp_K'].max()
    outlet_t = tdf['coolant_temperature_K'].iloc[-1]
    total_dp = cff['cumulative_pressure_drop_Pa'].iloc[-1]
    max_q = htf['coolant_side_htc_with_fins_W_per_m2K'].max() if 'coolant_side_htc_with_fins_W_per_m2K' in htf.columns else None
    rows.append({'variant': d, 'max_wall_temp_K': float(max_wt), 'coolant_outlet_temp_K': float(outlet_t), 'total_pressure_drop_Pa': float(total_dp), 'max_coolant_htc_W_per_m2K': float(max_q)})

out = pd.DataFrame(rows)
out.to_csv('gradient_sensitivity_summary.csv', index=False)
print(out)
