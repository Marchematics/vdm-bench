"""Build the deterministic VDM-Bench workloads from raw datasets."""
import argparse
import json
import os

import numpy as np

from vdmbench import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="workloads")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--quality-n", type=int, default=50_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("[1/3] analytics ...", flush=True)
    wa = core.build_analytics(a.data_root, n=a.n)
    np.savez_compressed(os.path.join(a.out, "analytics.npz"),
                        queries=wa["queries"], radii=wa["radii"], truth=wa["truth"],
                        anchors=wa["anchors"], r_cov=wa["r_cov"], k=wa["k"],
                        exact_covered=wa["exact_covered"], coverage_chosen=wa["coverage_chosen"],
                        X_join_n=wa["X_join_n"], r_join=wa["r_join"],
                        join_pairs=wa["join_pairs"], n=wa["n"])
    print("[2/3] quality ...", flush=True)
    wq = core.build_quality(a.data_root, n=a.quality_n)
    with open(os.path.join(a.out, "quality.json"), "w") as f:
        json.dump(wq, f, indent=2)
    print("[3/3] serving ...", flush=True)
    ws = core.build_serving()
    np.savez_compressed(os.path.join(a.out, "serving.npz"),
                        hard=ws["hard"], score=ws["score"], arrival=ws["arrival"])
    with open(os.path.join(a.out, "serving_meta.json"), "w") as f:
        json.dump({"tenants": ws["tenants"], "rho": ws["rho"], "cores": ws["cores"],
                   "surface": ws["surface"]}, f, indent=2)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"analytics_n": wa["n"], "quality_instances": len(wq["instances"]),
                   "serving_tenants": len(ws["tenants"])}, f, indent=2)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
