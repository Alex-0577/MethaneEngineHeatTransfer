import csv
import math
from pathlib import Path
fn = Path('AE-1310_processed_Plots/CSV_Data/temperature_distribution.csv')
cols = ['gas_temperature_K','gas_side_wall_temp_K','coolant_side_wall_temp_K','coolant_temperature_K']
if not fn.exists():
    print('MISSING', fn)
    raise SystemExit(1)
with fn.open(newline='') as f:
    reader=csv.DictReader(f)
    rows=list(reader)
print('rows_read', len(rows))
for c in cols:
    vals=[]
    for r in rows:
        v=r.get(c,'')
        try:
            vals.append(float(v))
        except:
            pass
    if vals:
        print(f'max {c} =', max(vals))
        print(f'min {c} =', min(vals))
    else:
        print('no numeric data for', c)
print('\nfirst 6 parsed rows:')
for r in rows[:6]:
    snippet={'axial_position_m': r.get('axial_position_m')}
    for k in cols:
        snippet[k]=r.get(k)
    print(snippet)
print('\nshow heat_transfer_parameters sample header and first line:')
fn2 = Path('AE-1310_processed_Plots/CSV_Data/heat_transfer_parameters.csv')
if fn2.exists():
    with fn2.open(newline='') as f:
        head = f.readline().strip()
        first = f.readline().strip()
    print('header:', head)
    print('first:', first)
else:
    print('missing', fn2)
