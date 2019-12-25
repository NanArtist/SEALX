'''For key generation from keylog.
   It can be useful if the server is down during the explaining process.
   In use, this file should be in the same directory with keylog to be processed.'''

import os, csv

filename = 'keylog'
data = list(csv.reader(open(filename, 'r')))

cands = {}
lCands = []  # [{(,),(,)...},{...},...]
candkeys = [row for row in data if row[0]!='False']
print('Remove {}/{} graphs whose pred_loss are more than 0.69.'.format(len(data)-len(candkeys), len(data)))
for candk in candkeys:
    cand = set(candk)
    if cand not in lCands:
        cands[len(lCands)] = 1
        lCands.append(cand)
    else:
        idx = lCands.index(cand)
        cands[idx] += 1
cands = sorted(cands.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
with open('keylog2keys', 'w', newline="") as f:
    writer = csv.writer(f)
    for key in cands:
        writer.writerow(list(lCands[key[0]])+[key[1]])
    writer.writerow(['Remove {}/{} graphs whose pred_loss are more than 0.69.'.format(len(data)-len(candkeys), len(data))])