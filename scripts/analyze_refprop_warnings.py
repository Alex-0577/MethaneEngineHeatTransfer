import csv
from collections import Counter
import os
p = 'refprop_warnings.csv'
if not os.path.exists(p):
    print('MISSING')
    raise SystemExit(1)

rows = []
with open(p, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for r in reader:
        if not r:
            continue
        rows.append(r)

print('total_rows:', len(rows))
# Determine header or not
first = rows[0]
has_header = False
if any('timestamp' in c.lower() for c in first) or any('refprop' in c.lower() for c in first):
    has_header = True
    header = [c.strip() for c in first]
    data = rows[1:]
else:
    header = None
    data = rows

# helpers to extract fields with safety
def safe_get(r, idx):
    return r[idx].strip() if idx < len(r) else ''

# Assume columns: 0 timestamp,1 type,6 herr, last message
types = [safe_get(r,1) for r in data]
herrs = [safe_get(r,6) for r in data]
messages = [safe_get(r,-1) for r in data]

c_type = Counter(types)
c_herr = Counter(herrs)
c_msg = Counter(messages)

print('\nTop 20 warning types:')
for k,v in c_type.most_common(20):
    print(f'{v:6d}  {k}')

print('\nTop 20 herr entries:')
for k,v in c_herr.most_common(20):
    print(f'{v:6d}  {k}')

print('\nTop 20 messages (truncated to 120 chars):')
for k,v in c_msg.most_common(20):
    print(f'{v:6d}  {k[:120]}')

print('\nFirst 5 data rows:')
for r in data[:5]:
    print(','.join(r))

print('\nLast 5 data rows:')
for r in data[-5:]:
    print(','.join(r))
