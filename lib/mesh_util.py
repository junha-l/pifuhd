"""
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid


def reconstruction(
    net,
    cuda,
    calib_tensor,
    resolution,
    thresh=0.5,
    num_samples=10000,
):
    # Create grid points
    coords, mat = create_grid(resolution, resolution, resolution)
    calib = calib_tensor[0].cpu().numpy()
    calib_inv = np.linalg.inv(calib)
    coords = coords.reshape(3, -1).T
    coords = np.matmul(
        np.concatenate([coords, np.ones((coords.shape[0], 1))], 1), calib_inv.T
    )[:, :3]
    coords = coords.T.reshape(3, resolution, resolution, resolution)

    def eval_func(points):
        ########## TODO: Eval occupancy fields ##########
        pred = None
        ########## [End] Eval occupancy fields ##########
        return pred.detach().cpu().numpy()

    sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, thresh)
        trans_mat = np.matmul(calib_inv, mat)
        verts = np.matmul(trans_mat[:3, :3], verts.T) + trans_mat[:3, 3:4]
        verts = verts.T
        if np.linalg.det(trans_mat[:3, :3]) < 0.0:
            faces = faces[:, ::-1]
        return verts, faces, normals, values
    except:
        print("error cannot marching cubes")
        return -1


def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, "w")

    for v in verts:
        file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    if faces is not None:
        for f in faces:
            if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
                continue
            f_plus = f + 1
            file.write("f %d %d %d\n" % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, "w")

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write(
            "v %.4f %.4f %.4f %.4f %.4f %.4f\n" % (v[0], v[1], v[2], c[0], c[1], c[2])
        )
    for f in faces:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, "w")

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        file.write("vt %.4f %.4f\n" % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write(
            "f %d/%d %d/%d %d/%d\n"
            % (f_plus[0], f_plus[0], f_plus[2], f_plus[2], f_plus[1], f_plus[1])
        )
    file.close()
