import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = os.path.abspath(os.path.dirname(__file__))
pre_csv_dir = os.path.join(root, 'AE-1310_pre_processed_Plots', 'CSV_Data')
post_csv_dir = os.path.join(root, 'AE-1310_post_processed_Plots', 'CSV_Data')
out_dir = os.path.join(root, 'AE-1310_comparison')
os.makedirs(out_dir, exist_ok=True)

# helper to load CSVs
def load_temperature(path):
    return pd.read_csv(path)

def load_flow(path):
    return pd.read_csv(path)

def load_heat(path):
    return pd.read_csv(path)

pre_temp = load_temperature(os.path.join(pre_csv_dir, 'temperature_distribution.csv'))
post_temp = load_temperature(os.path.join(post_csv_dir, 'temperature_distribution.csv'))

pre_flow = load_flow(os.path.join(pre_csv_dir, 'coolant_flow_parameters.csv'))
post_flow = load_flow(os.path.join(post_csv_dir, 'coolant_flow_parameters.csv'))

pre_heat = load_heat(os.path.join(pre_csv_dir, 'heat_transfer_parameters.csv'))
post_heat = load_heat(os.path.join(post_csv_dir, 'heat_transfer_parameters.csv'))

# Ensure axial position alignment (interpolate post onto pre if necessary)
ax_pre = pre_temp['axial_position_m']
ax_post = post_temp['axial_position_m']

# Plot temperatures: gas wall and coolant temp
plt.figure(figsize=(8,4))
plt.plot(ax_pre, pre_temp['gas_side_wall_temp_K'], label='pre gas wall')
plt.plot(ax_post, post_temp['gas_side_wall_temp_K'], label='post gas wall')
plt.plot(ax_pre, pre_temp['coolant_temperature_K'], '--', label='pre coolant')
plt.plot(ax_post, post_temp['coolant_temperature_K'], '--', label='post coolant')
plt.xlabel('axial_position_m')
plt.ylabel('Temperature (K)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'temperature_comparison.png'))
plt.close()

# Plot coolant-side HTC
plt.figure(figsize=(8,4))
plt.plot(pre_heat['axial_position_m'], pre_heat['coolant_side_htc_with_fins_W_per_m2K'], label='pre h_c')
plt.plot(post_heat['axial_position_m'], post_heat['coolant_side_htc_with_fins_W_per_m2K'], label='post h_c')
plt.xlabel('axial_position_m')
plt.ylabel('h_c (W/m2K)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'hc_comparison.png'))
plt.close()

# Build summary metrics
def summarize(temp_df, flow_df, heat_df):
    d = {}
    d['max_wall_temp_K'] = float(temp_df['gas_side_wall_temp_K'].max())
    d['max_coolant_temp_K'] = float(temp_df['coolant_temperature_K'].max())
    d['coolant_T_in_K'] = float(temp_df['coolant_temperature_K'].iloc[0])
    d['coolant_T_out_K'] = float(temp_df['coolant_temperature_K'].iloc[-1])
    d['coolant_total_temp_rise_K'] = d['coolant_T_out_K'] - d['coolant_T_in_K']
    # final cumulative pressure drop if exists
    if 'cumulative_pressure_drop_Pa' in flow_df.columns:
        d['final_pressure_drop_Pa'] = float(flow_df['cumulative_pressure_drop_Pa'].iloc[-1])
    elif 'pressure_drop_Pa' in flow_df.columns:
        d['final_pressure_drop_Pa'] = float(flow_df['pressure_drop_Pa'].cumsum().iloc[-1])
    else:
        d['final_pressure_drop_Pa'] = float('nan')
    # max h_c
    if 'coolant_side_htc_with_fins_W_per_m2K' in heat_df.columns:
        d['max_hc_W_per_m2K'] = float(heat_df['coolant_side_htc_with_fins_W_per_m2K'].max())
    else:
        d['max_hc_W_per_m2K'] = float('nan')
    return d

summary_pre = summarize(pre_temp, pre_flow, pre_heat)
summary_post = summarize(post_temp, post_flow, post_heat)

summary_df = pd.DataFrame([summary_pre, summary_post], index=['pre_mdot_1.56','post_mdot_3.12'])
summary_df.to_csv(os.path.join(out_dir, 'summary_table.csv'))

# Also save a small CSV with axial comparison (sampled to common grid)
common_ax = sorted(set(ax_pre.round(6)).union(set(ax_post.round(6))))
common_ax = pd.Series(common_ax)
pre_interp = pd.DataFrame({
    'axial_position_m': common_ax,
    'pre_gas_wall_K': pd.Series(np.interp(common_ax, ax_pre, pre_temp['gas_side_wall_temp_K'])),
    'pre_coolant_K': pd.Series(np.interp(common_ax, ax_pre, pre_temp['coolant_temperature_K']))
})
post_interp = pd.DataFrame({
    'axial_position_m': common_ax,
    'post_gas_wall_K': pd.Series(np.interp(common_ax, ax_post, post_temp['gas_side_wall_temp_K'])),
    'post_coolant_K': pd.Series(np.interp(common_ax, ax_post, post_temp['coolant_temperature_K']))
})
axial_compare = pre_interp.merge(post_interp, on='axial_position_m')
axial_compare.to_csv(os.path.join(out_dir, 'axial_temperature_comparison.csv'), index=False)

print('comparison generated in', out_dir)
