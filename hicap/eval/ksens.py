"""K-sensitivity: fixed-K methods need the cluster count; ours does not.
Sweep K = factor * C_true and report MoF. Shows the count-free advantage:
TW-FINCH must be handed the right K; our one hierarchy is read at any level.

    python -m hicap.eval.ksens [dataset] [data_root]
    (dataset default 50salads; data_root default the local 50 Salads path)
"""
import json
import os
import sys

import numpy as np

from hicap.data import tas
from hicap.eval import baselines
from hicap.eval.tas_metrics import mof_hungarian
from hicap.segment.hierarchy import merge_order, segmentation_at_k

dataset = sys.argv[1] if len(sys.argv) > 1 else "50salads"
root = sys.argv[2] if len(sys.argv) > 2 else "/home/chaddy/datasets/50_salads/data"
factors = [0.5, 0.75, 1.0, 1.5, 2.0]

vids = tas.list_videos(root, dataset)
acc = {m: {f: [] for f in factors} for m in ["twfinch", "kmeans", "ours_contig"]}
for vid in vids:
    F = tas.load_features(root, dataset, vid)
    gt = tas.load_labels(root, dataset, vid)
    C = len(set(gt.tolist()))
    order = merge_order(F, linkage="ward", l2_normalize=False)
    for f in factors:
        K = max(2, int(round(f * C)))
        acc["twfinch"][f].append(mof_hungarian(baselines.twfinch_labels(F, K, 10), gt)[0] * 100)
        acc["kmeans"][f].append(mof_hungarian(baselines.kmeans_labels(F, K, 10), gt)[0] * 100)
        acc["ours_contig"][f].append(mof_hungarian(segmentation_at_k(order, len(F), K), gt)[0] * 100)

print(f"{dataset}: K-sensitivity (MoF, mean over {len(vids)} videos)")
print(f'{"K = factor*C_true":<18}' + "".join(f'{f"x{f}":>9}' for f in factors))
res = {}
for m in ["twfinch", "kmeans", "ours_contig"]:
    row = [float(np.mean(acc[m][f])) for f in factors]
    res[m] = dict(zip(map(str, factors), row))
    print(f"{m:<18}" + "".join(f"{v:>9.1f}" for v in row))
print()
for m in ["twfinch", "kmeans", "ours_contig"]:
    vals = list(res[m].values())
    print(f"{m}: best MoF {max(vals):.1f} at x{factors[int(np.argmax(vals))]}, "
          f"drop to {min(vals):.1f} (range {max(vals) - min(vals):.1f})")
os.makedirs("hicap/results", exist_ok=True)
json.dump(res, open(f"hicap/results/ksens_report_{dataset}.json", "w"), indent=2)
