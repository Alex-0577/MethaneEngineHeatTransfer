import csv
from pathlib import Path
p=Path('AE-1310_processed_Plots/CSV_Data')
cp=p/'coolant_properties.csv'
if not cp.exists():
    print('missing',cp); raise SystemExit(1)
with cp.open(newline='') as f:
    r=csv.DictReader(f)
    rows=list(r)
bad=[]
for i,row in enumerate(rows):
    try:
        mu=float(row.get('coolant_viscosity_Pa_s',''))
    except:
        mu=None
    try:
        k=float(row.get('coolant_conductivity_W_per_mK',''))
    except:
        k=None
    if mu is None or k is None:
        bad.append((i,row))
        continue
    if mu<=0 or k<=0 or mu<-1 or k<-1e5:
        bad.append((i,row))

print('bad count',len(bad))
for i,row in bad[:20]:
    print(i, row.get('axial_position_m'), row.get('coolant_temperature_K',''), 'mu=',row.get('coolant_viscosity_Pa_s'), 'k=',row.get('coolant_conductivity_W_per_mK'))
