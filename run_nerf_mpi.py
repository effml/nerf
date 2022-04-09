import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
from run_nerf_helpers import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

import run_nerf

tf.compat.v1.enable_eager_execution()

def render_planes(H, W, focal, mpi=False,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           plane_depths=None, num_planes=12, disparity=False, 
           min_depth=0.0, max_depth=1.0,use_viewdirs=False, 
           c2w_staticcam=None, **kwargs,
           ):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      plane_depths: list. If not None, a list of depth values to evaluate the MPI at.
      num_planes: int. The number of planes to evaluate. Used only if plane_depths is None.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.

    Returns:
      result: (num_planes, H, W, 4) images of each plane
    """

    if plane_depths is None:
        if disparity:
            plane_depths = 1.0 / np.linspace(
                1.0 / (min_depth + 1e-8), 
                1.0 / max_depth, 
                num_planes,
            )
        else:
            plane_depths = np.linspace(min_depth, max_depth, num_planes)
    else:
        num_planes = len(plane_depths)
    print("NUM_PLANES:", num_planes)
    print("DEPTHS:", plane_depths)
    result_layers = []
    for i in range(num_planes - 1):
        print("Layer", i)
        near_depth = plane_depths[i]
        far_depth = plane_depths[i + 1]

        rgb, _, acc, _ = run_nerf.render(
            H, W, focal, chunk=chunk, rays=rays, c2w=c2w, 
            ndc=ndc, near=near_depth, far=far_depth, mpi_depth=mpi,
            use_viewdirs=use_viewdirs, c2w_staticcam=c2w_staticcam, **kwargs,
        )
        acc = tf.expand_dims(acc, axis=-1)
        result_layers.append(tf.concat((rgb, acc), axis=-1))
    
    full_rgb = run_nerf.render(
        H, W, focal, chunk=chunk, rays=rays, c2w=c2w, 
        ndc=ndc, near=0, far=1.0, mpi_depth=mpi, 
        use_viewdirs=use_viewdirs, c2w_staticcam=c2w_staticcam, **kwargs,
    )[0]

    final_layer = tf.concat((full_rgb, tf.ones((H, W, 1))), axis=-1)

    result_layers.append(final_layer) # only the rgb, full image at the very end
    
    return tf.stack(result_layers)