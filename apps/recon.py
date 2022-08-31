# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm

from lib.options import BaseOptions
from lib.mesh_util import save_obj_mesh, reconstruction
from lib.data import EvalWPoseDataset
from lib.model import PIFuCoarse, PIFuFine


parser = BaseOptions()


def gen_mesh(res, net, cuda, data, save_path, thresh=0.5):
    image_tensor_global = data["img_512"].to(device=cuda)
    image_tensor = data["img"].to(device=cuda)
    calib_tensor = data["calib"].to(device=cuda)

    # Run image encoders of coarse & fine model
    net.filter_global(image_tensor_global)
    net.filter_local(image_tensor[:, None])

    try:
        verts, faces, _, _ = reconstruction(
            net,
            cuda,
            calib_tensor,
            res,
            thresh,
            num_samples=50000,
        )
        save_obj_mesh(save_path, verts, faces)
    except Exception as e:
        print(e)


def recon(opt):
    cuda = torch.device("cuda:%d" % opt.gpu_id if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    state_dict_path = opt.load_netMR_checkpoint_path

    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        print("Resuming from ", state_dict_path)
        state_dict = torch.load(state_dict_path, map_location=cuda)
        print("Warning: opt is overwritten.")
        dataroot = opt.dataroot
        resolution = opt.resolution
        results_path = opt.results_path
        loadSize = opt.loadSize

        opt = state_dict["opt"]
        opt.dataroot = dataroot
        opt.resolution = resolution
        opt.results_path = results_path
        opt.loadSize = loadSize
    else:
        raise Exception("failed loading state dict!", state_dict_path)

    ## Prepare dataset
    test_dataset = EvalWPoseDataset(opt)
    print("test data size: ", len(test_dataset))
    projection_mode = test_dataset.projection_mode

    ## Prepare model
    opt_netG = state_dict["opt_netG"]
    net_coarse = PIFuCoarse(opt_netG, projection_mode).to(device=cuda)
    net_fine = PIFuFine(opt, net_coarse, projection_mode).to(device=cuda)

    def set_eval():
        net_coarse.eval()

    # load checkpoints
    net_fine.load_state_dict(state_dict["model_state_dict"])

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs("%s/%s/recon" % (opt.results_path, opt.name), exist_ok=True)

    start_id, end_id = 0, len(test_dataset)

    ## test
    with torch.no_grad():
        set_eval()

        print("generate mesh (test) ...")
        for i in tqdm(range(start_id, end_id)):
            if i >= len(test_dataset):
                break

            test_data = test_dataset[i]
            save_path = "%s/%s/recon/result_%s_%d.obj" % (
                opt.results_path,
                opt.name,
                test_data["name"],
                opt.resolution,
            )
            print(save_path)
            gen_mesh(
                opt.resolution,
                net_fine,
                cuda,
                test_data,
                save_path,
            )


def reconWrapper(args=None):
    opt = parser.parse(args)
    recon(opt)


if __name__ == "__main__":
    reconWrapper()
