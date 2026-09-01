#!/usr/bin/env python3
"""Run vendored RigNet inference on one of our own generated meshes.

Not part of RigNet itself -- glue between our repo and the vendored, macOS-patched
checkout at vendor/rignet (see scripts/patch_rignet_macos_compat.py). Drives the same
five-network pipeline quick_start.py's __main__ block runs, just parameterized on our
own model_id instead of RigNet's bundled examples.

Usage (run with vendor/rignet's own venv, which has the patched torch_geometric stack):
    vendor/rignet/.venv/bin/python scripts/rignet_infer.py <model_id>

Expects vendor/rignet/quick_start/<model_id>_ori.obj to already exist.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RIGNET = ROOT / "vendor" / "rignet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="basename of quick_start/<model_id>_ori.obj")
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()

    os.chdir(RIGNET)
    sys.path.insert(0, str(RIGNET))

    import numpy as np
    import open3d as o3d
    import torch

    import quick_start
    from quick_start import (
        JOINTNET,
        ROOTNET,
        BONENET,
        SKINNET,
        create_single_data,
        predict_joints,
        predict_skeleton,
        predict_skinning,
        tranfer_to_ori_mesh,
    )

    input_folder = "quick_start/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # predict_skinning does `global device` -- it assumes quick_start.py's own
    # __main__ block set this at module scope, which never runs when we import its
    # functions instead. Supply it directly rather than patching vendored code for
    # what's really a caller contract.
    quick_start.device = device

    print("loading all networks...")
    jointNet = JOINTNET()
    jointNet.to(device)
    jointNet.eval()
    jointNet.load_state_dict(torch.load("checkpoints/gcn_meanshift/model_best.pth.tar", map_location=device)["state_dict"])

    rootNet = ROOTNET()
    rootNet.to(device)
    rootNet.eval()
    rootNet.load_state_dict(torch.load("checkpoints/rootnet/model_best.pth.tar", map_location=device)["state_dict"])

    boneNet = BONENET()
    boneNet.to(device)
    boneNet.eval()
    boneNet.load_state_dict(torch.load("checkpoints/bonenet/model_best.pth.tar", map_location=device)["state_dict"])

    skinNet = SKINNET(nearest_bone=5, use_Dg=True, use_Lf=True)
    skinNet.load_state_dict(torch.load("checkpoints/skinnet/model_best.pth.tar", map_location=device)["state_dict"])
    skinNet.to(device)
    skinNet.eval()
    print("     all networks loaded.")

    model_id = args.model_id
    mesh_filename = os.path.join(input_folder, f"{model_id}_remesh.obj")
    if not os.path.exists(mesh_filename):
        mesh_ori_filename = os.path.join(input_folder, f"{model_id}_ori.obj")
        mesh_ori = o3d.io.read_triangle_mesh(mesh_ori_filename)
        if len(np.asarray(mesh_ori.vertices)) == 0:
            print(f"Please name your input model as {model_id}_ori.obj")
            sys.exit(1)
        mesh_remesh = mesh_ori.simplify_quadric_decimation(4000)
        o3d.io.write_triangle_mesh(mesh_filename, mesh_remesh)
        print(f"decimated to {len(np.asarray(mesh_remesh.vertices))} vertices")

    data, vox, surface_geodesic, translation_normalize, scale_normalize = create_single_data(mesh_filename)
    data.to(device)

    print("predicting joints")
    data = predict_joints(
        data, vox, jointNet, args.threshold, bandwidth=args.bandwidth,
        mesh_filename=mesh_filename.replace("_remesh.obj", "_normalized.obj"),
    )
    data.to(device)

    print("predicting connectivity")
    pred_skeleton = predict_skeleton(
        data, vox, rootNet, boneNet,
        mesh_filename=mesh_filename.replace("_remesh.obj", "_normalized.obj"),
    )

    print("predicting skinning")
    pred_rig = predict_skinning(
        data, pred_skeleton, skinNet, surface_geodesic,
        mesh_filename.replace("_remesh.obj", "_normalized.obj"),
        subsampling=True,
    )

    pred_rig.normalize(scale_normalize, -translation_normalize)

    print("saving result")
    mesh_filename_ori = os.path.join(input_folder, f"{model_id}_ori.obj")
    pred_rig = tranfer_to_ori_mesh(mesh_filename_ori, mesh_filename, pred_rig)
    out_path = mesh_filename_ori.replace(".obj", "_rig.txt")
    pred_rig.save(out_path)
    print(f"done: {out_path}")


if __name__ == "__main__":
    main()
