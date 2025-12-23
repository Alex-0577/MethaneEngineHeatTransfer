import csv,math,statistics
from pathlib import Path
p=Path('AE-1310_processed_Plots/CSV_Data')
ht= p/'heat_transfer_parameters.csv'
cp= p/'coolant_properties.csv'
td= p/'temperature_distribution.csv'
if not ht.exists() or not cp.exists() or not td.exists():
    print('missing one of files', ht.exists(), cp.exists(), td.exists())
    raise SystemExit(1)

# helper to parse possible complex string like '(1359.6485+0j)'
def parse_real(s):
    if s is None: return None
    s=str(s).strip()
    if s=='' : return None
    # remove parentheses
    s=s.replace('(','').replace(')','')
    # handle complex
    if '+' in s and 'j' in s:
        try:
            return float(complex(s).real)
        except:
            pass
    # plain float
    try:
        return float(s)
    except:
        # sometimes broken spacing/newline; strip non-numeric
        import re
        m=re.search(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", s)
        if m:
            return float(m.group(0))
    return None

# read files
with ht.open(newline='') as f:
    r=csv.DictReader(f)
    ht_rows=list(r)
with cp.open(newline='') as f:
    r=csv.DictReader(f)
    cp_rows=list(r)
with td.open(newline='') as f:
    r=csv.DictReader(f)
    td_rows=list(r)

n=len(ht_rows)
print('rows', n)
# compute h_c stats
hcs=[]
for row in ht_rows:
    val=parse_real(row.get('coolant_side_htc_with_fins_W_per_m2K'))
    if val is not None:
        hcs.append(val)
print('h_c: count',len(hcs),'mean',statistics.mean(hcs),'max',max(hcs),'min',min(hcs))

# coolant properties stats (use cp_rows)
rhos=[]; mus=[]; cps=[]; ks=[]
for row in cp_rows:
    try:
        rhos.append(float(row.get('coolant_density_kg_per_m3','')))
    except: pass
    try:
        mus.append(float(row.get('coolant_viscosity_Pa_s','')))
    except: pass
    try:
        cps.append(float(row.get('coolant_specific_heat_J_per_kgK','')))
    except: pass
    try:
        ks.append(float(row.get('coolant_conductivity_W_per_mK','')))
    except: pass
if rhos: print('rho mean',statistics.mean(rhos),'range',min(rhos),max(rhos))
if mus: print('mu mean',statistics.mean(mus),'range',min(mus),max(mus))
if cps: print('cp mean',statistics.mean(cps),'range',min(cps),max(cps))
if ks: print('k mean',statistics.mean(ks),'range',min(ks),max(ks))

# compute per-segment heat removed by coolant using delta T from temperature_distribution (coolant_temperature column increments)
# use cp from cp_rows per-row
q_per_seg=[]
m_dot = 0.78  # from parameters file; if differs, user can change
for i,row in enumerate(td_rows):
    try:
        T_curr=float(row.get('coolant_temperature_K',''))
    except: 
        continue
    if i==0:
        continue
    try:
        T_prev=float(td_rows[i-1].get('coolant_temperature_K',''))
    except:
        continue
    dT=T_curr-T_prev
    if i-1 < len(cp_rows):
        try:
            cp_loc=float(cp_rows[i-1].get('coolant_specific_heat_J_per_kgK'))
        except:
            cp_loc=3400
    else:
        cp_loc=3400
    q = m_dot * cp_loc * dT
    q_per_seg.append(q)
print('q_per_seg: count',len(q_per_seg),'mean',statistics.mean(q_per_seg) if q_per_seg else None,'max',max(q_per_seg) if q_per_seg else None,'min',min(q_per_seg) if q_per_seg else None)

# sample problematic rows where gas_side_wall_temp is very low relative to gas temp
samples=[]
for i,row in enumerate(td_rows[:20]):
    gw=parse_real(row.get('gas_side_wall_temp_K'))
    gc=parse_real(row.get('coolant_temperature_K'))
    gt=parse_real(row.get('gas_temperature_K'))
    samples.append((i,gt,gw,gc))
print('\nfirst 10 samples (index, gasT, wallT_gas_side, coolantT):')
for s in samples[:10]:
    print(s)

# print top 5 largest h_c rows
top_hc = sorted([(parse_real(r.get('coolant_side_htc_with_fins_W_per_m2K')),i) for i,r in enumerate(ht_rows)], key=lambda x: (x[0] if x[0] is not None else -1), reverse=True)[:5]
print('\ntop 5 h_c values (value, index):')
print(top_hc)

# finish
print('\nDone')
