import csv,statistics
from pathlib import Path
fn = Path('AE-1310_processed_Plots/CSV_Data/temperature_distribution.csv')
with fn.open(newline='') as f:
    reader=csv.DictReader(f)
    rows=list(reader)
print('rows',len(rows))
for col in ['gas_side_wall_temp_K','coolant_temperature_K','coolant_side_wall_temp_K','gas_temperature_K']:
    vals=[]
    for r in rows:
        try:
            vals.append(float(r.get(col,'')))
        except:
            pass
    if vals:
        print(col, 'count', len(vals), 'mean', statistics.mean(vals), 'max', max(vals), 'min', min(vals))
    else:
        print('no data for', col)

print('\nfirst 6 rows sample:')
for r in rows[:6]:
    print({k:r.get(k,'') for k in ['axial_position_m','gas_temperature_K','gas_side_wall_temp_K','coolant_side_wall_temp_K','coolant_temperature_K']})
