'''For key generation from keylog.
   It can be useful if the server is down during the explaining process.
   There is no need to move the file.'''

import os, csv
import sys
sys.path.append('%s/../../' % os.path.dirname(os.path.realpath(__file__)))
from explain_main import arg_parse
from utils import io_utils

args = arg_parse()
data = list(csv.reader(open(os.path.join(args.logdir, io_utils.gen_explainer_prefix(args), 'keylog'), 'r')))

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
with open(os.path.join(args.logdir, io_utils.gen_explainer_prefix(args), 'keylog2keys'), 'w', newline="") as f:
    writer = csv.writer(f)
    for key in cands:
        writer.writerow(list(lCands[key[0]])+[key[1]])
    writer.writerow(['Remove {}/{} graphs whose pred_loss are more than 0.69.'.format(len(data)-len(candkeys), len(data))])