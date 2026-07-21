"""K-sensitivity: fixed-K methods need the cluster count; ours does not.
Sweep K = factor * C_true and report MoF. Shows the count-free advantage:
TW-FINCH must be handed the right K; our one hierarchy is read at any level."""
import json
import numpy as np
from hicap.data import salads
from hicap.eval import baselines
from hicap.eval.tas_metrics import mof_hungarian
from hicap.segment.hierarchy import merge_order, segmentation_at_k

root = '/home/chaddy/datasets/50_salads/data'
vids = salads.list_videos(root)
factors = [0.5, 0.75, 1.0, 1.5, 2.0]

acc = {m: {f: [] for f in factors} for m in ['twfinch', 'kmeans', 'ours_contig']}
for vid in vids:
    F = salads.load_features(root, vid)
    gt = salads.load_labels(root, vid, 'groundTruth')
    C = len(set(gt.tolist()))
    order = merge_order(F, linkage='ward', l2_normalize=False)
    for f in factors:
        K = max(2, int(round(f * C)))
        acc['twfinch'][f].append(mof_hungarian(baselines.twfinch_labels(F, K, 10), gt)[0] * 100)
        acc['kmeans'][f].append(mof_hungarian(baselines.kmeans_labels(F, K, 10), gt)[0] * 100)
        acc['ours_contig'][f].append(mof_hungarian(segmentation_at_k(order, len(F), K), gt)[0] * 100)

print(f'{"K = factor*C_true":<18}' + ''.join(f'{f"x{f}":>9}' for f in factors))
res = {}
for m in ['twfinch', 'kmeans', 'ours_contig']:
    row = [float(np.mean(acc[m][f])) for f in factors]
    res[m] = dict(zip(map(str, factors), row))
    print(f'{m:<18}' + ''.join(f'{v:>9.1f}' for v in row))
# degradation from best to worst across K
print()
for m in ['twfinch', 'kmeans', 'ours_contig']:
    vals = list(res[m].values())
    print(f'{m}: best MoF {max(vals):.1f} at x{factors[int(np.argmax(vals))]}, '
          f'drop to {min(vals):.1f} (range {max(vals)-min(vals):.1f})')
json.dump(res, open('hicap/results/ksens_report.json', 'w'), indent=2)
