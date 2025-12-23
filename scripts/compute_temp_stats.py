import csv
from pathlib import Path
fn = Path('AE-1310_processed_Plots/CSV_Data/temperature_distribution.csv')
cols=['gas_side_wall_temp_K','coolant_temperature_K']
with fn.open(newline='') as f:
    reader=csv.DictReader(f)
    rows=list(reader)
for c in cols:
    vals=[]
    for r in rows:
        try:
            vals.append(float(r.get(c,'')))
        except:
            pass
    if vals:
        import statistics
        print(c, 'count', len(vals), 'mean', statistics.mean(vals), 'max', max(vals), 'min', min(vals))
    else:
        print('no data for', c)
