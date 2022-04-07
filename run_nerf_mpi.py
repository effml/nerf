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

def render_planes(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           plane_depths=None, num_planes=12,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
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
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if plane_depths is None:
        plane_depths = np.linspace(0.0, 1.0, num_planes)
    else:
        num_planes = len(plane_depths)
    for i in range(num_planes - 1):
        near_depth = plane_depths[i]
        far_depth = plane_depths[i + 1]

        plane_outputs = run_nerf.render(
            H, W, focal, chunk=chunk, rays=rays, c2w=c2w, 
            ndc=ndc, near=near_depth, far=far_depth, 
            use_viewdirs=use_viewdirs, c2w_staticcam=c2w_staticcam,
        )
    
        # do something with this
