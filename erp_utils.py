import math
import os
import tensorflow as tf
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.enable_eager_execution()

# Tensorflow version of https://github.com/effml/360stereo/blob/main/tools/equirectangular.ipynb

DEFAULT_POSE = tf.eye(4)


def normuv(uv, H, W):
    """
    Normalize pixel coordinates to lie in [-1, 1]
    
    :param uv (..., 2) unnormalized pixel coordinates for HxW image
    """
    u = uv[..., 0] / H * 2.0 - 1.0
    v = uv[..., 1] / W * 2.0 - 1.0
    return tf.stack([u, v], axis=-1)


def unnormuv(uv, H, W):
    """
    Un-normalize pixel coordinates
    
    :param uv (..., 2) normalized pixel coordinates in [-1, 1]
    """
    u = (uv[..., 0] + 1.0) / 2.0 * H
    v = (uv[..., 1] + 1.0) / 2.0 * W
    return tf.stack([u, v], axis=-1)
    

def get_uv(H, W):
    """
    Get normalized uv coordinates for image

    :param height (int) source image height
    :param width (int) source image width
    :return uv (N, 2) pixel coordinates in [-1.0, 1.0]
    """
    yy, xx = tf.meshgrid(
        (tf.arange(H, dtype=tf.float32) + 0.5),
        (tf.arange(W, dtype=tf.float32) + 0.5),
    )
    uv = tf.stack([xx, yy], axis=-1) # (H, W, 2)
    uv = normuv(uv, W, H) # (H, W, 2)
    return tf.reshape(uv, (H * W, 2))


def get_r(uv, pose=DEFAULT_POSE, radii=tf.Tensor([1.])):
    """
    Converts list of float radii to a per-pixel radii.
    
    Assumes that there are multiple spherical images at the specific float radii.
    Then, computes distance from the provided camera pose to the spherical images
    
    :param uv (N, 2)
    :param poses (4, 4)
    :param radii (R,)
    :return r (N, R)
    
    >>> uv = get_uv(8, 8)
    >>> R = get_r(uv)
    >>> np.allclose(R, 1)
    True
    >>> pose = torch.eye(4)
    >>> pose[1, 3] = 0.5 # shift
    >>> R = get_r(uv, pose)
    >>> (R.min() - 0.5).abs() < 0.1
    tensor(True)
    >>> (R.max() - 1.5).abs() < 0.1
    tensor(True)
    >>> R = get_r(uv, pose, radii=torch.Tensor([1, 2]))
    >>> (R.max() - 2.5).abs() < 0.1
    tensor(True)
    """
    N, _ = uv.shape
    
    R = radii.shape[-1]
    uvr = build_uvr(uv, tf.repeat(tf.reshape(radii, (1, R)), repeats=(N, 1), axis=0))
    rays = erp_rays(uvr, pose) # cast rays using camera poses
    rays = tf.reshape(rays, (N, R, 7))
    O, D, t = rays[..., :3], rays[..., 3:6], rays[..., 6:] # (N, R, 3), (N, R, 3), (N, R, 1)
    
    # compute quadratic form coefficients
    a = tf.reshape(tf.reduce_sum(D ** 2., axis=-1), (N, R))
    b = tf.reshape(tf.reduce_sum(2 * (D * O), axis=-1), (N, R))
    c = tf.reshape(tf.reduce_sum(O ** 2., axis=-1), (N, R)) - radii[None] ** 2.
    
    # assert at least one solution (we are in the smallest sphere)
    discriminant = b ** 2. - 4 * a * c # (N, R)
    # assert (discriminant >= 0).all(), (
    #     f'{(discriminant < 0).sum()} / {np.prod(discriminant.shape)} rays do not intersect the sphere.\n'
    #     f'Min radius: {radii.min()}\n'
    #     f'Pose:\n{pose}'
    # )
    
    # solve for ray intersection with sphere
    t0 = (-b + tf.math.sqrt(discriminant)) / (2. * a) # (N, R) - TODO: check all solns? check sgn of t0?
    intersection = O + D * t0[:, :, None] # (N, R, 3)
    per_pixel_radii = tf.norm(intersection - pose[None, None, :3, 3], axis=-1) # (N, R)
    return tf.reshape(per_pixel_radii, (N, R))


def build_uvr(uv, r):
    """
    Assemble spherical coordinates for image from uv and radii
    
    :param uv (N, 2) pixel coordinates in [-1.0, 1.0]
    :param r (N, R) radii per pixel
    :return uvr (N, R, 3) pixel coordinates in [-1.0, 1.0] + radii
    """
    R = r.shape[-1]
    N, _ = uv.shape
    return tf.cat([
        tf.repeat(tf.reshape(uv, (N, 1, 2)), repeats = (1, R, 1)), # (N, R, 2)
        tf.reshape(r, (N, R, 1)), # (N, R, 1)
    ], axis=-1)


def get_uvr(H, W, pose=DEFAULT_POSE, radii=torch.Tensor([1])):
    """
    Get normalized uv coordinates + radius for image. Convenience wrapper
    for both `build_uvr` and `get_uv`.

    :param height (int) source image height
    :param width (int) source image width
    :param radii [R] radii of the spheres
    :return uvr (H, W, R, 3) pixel coordinates in [-1.0, 1.0] + radii
    """
    R = radii.shape[-1]
    uv = get_uv(H, W) # (N, 2)
    return build_uvr(
        uv=uv,
        r=get_r(uv, pose, radii=radii),
    )


def erp2xyz(uvr, rhs=True):
    """
    Convert equirectangular pixel coordinate to camera-space cartesian
    coordinates.
    
    :param uvr (..., 3) pixel coordinates in [-1.0, 1.0] + radii
    :return xyz (..., 3) cartesian coordinates
    
    >>> xyz = torch.rand(1000, 3) * 2 - 1
    >>> xyz2 = erp2xyz(xyz2erp(xyz))
    >>> np.allclose(xyz, xyz2, atol=1e-4)
    True
    """
    lon = uvr[..., 0] * math.pi
    lat = uvr[..., 1] * (math.pi * 0.5)
    if not rhs:
      lat = -lat
    radius = uvr[..., 2]
    return tf.stack([
        radius * tf.cos(lat) * tf.sin(lon),
        radius * tf.sin(lat),
        radius * tf.cos(lat) * tf.cos(lon),
    ], axis=-1)


def xyz2erp(xyz, rhs=True):
    """
    Convert camera-space cartesian coordinates to equirectangular pixel
    coordinate.
    
    :param xyz (..., 3) cartesian coordinates
    :return uvr (..., 3) pixel coordinates in [-1.0, 1.0] + radii
    
    >>> uv = get_uvr(2048, 1024)
    >>> uv2 = xyz2erp(erp2xyz(uv))
    >>> np.allclose(uv, uv2, atol=1e-4)
    True
    >>> uv3 = get_uvr(8, 8, radii=torch.Tensor([5, 10, 100, 200]))
    >>> uv4 = xyz2erp(erp2xyz(uv3))
    >>> np.allclose(uv3, uv4)
    True
    """
    # TODO: account for pixel shift
    radii = tf.norm(xyz, axis=-1)
    xyz = xyz / radii[..., None]
    lat = tf.asin(tf.clip_by_value(xyz[..., 1], -1.0, 1.0))
    lon = tf.atan2(xyz[..., 0], xyz[..., 2])
    if not rhs:
      lat = -lat
    return tf.stack([
        lon / math.pi,
        2.0 * lat / math.pi,
        radii,
    ], axis=-1)


def erp_rays(uvr, pose=DEFAULT_POSE):
    """
    Generate ray bundle for single spherical camera.
    
    :param uvr [N, R, 3]
    :param poses [3, 4]
    :param radii [R]
    :return rays [N, 7]
    """
    return erp_rays_batched(uvr[None], pose[None])[0]


def erp_rays_batched(uvrs, poses=DEFAULT_POSE[None]):
    """
    Generate ray bundle for batch of spherical cameras.
    
    :param uvrs [B, N, R, 3]
    :param poses [B, 3, 4]
    :param radii [R]
    :return rays [B, N * R, 7], last channel: origins (3), directions (3), length (3)
    """
    B, N, R, _ = uvrs.shape
    
    # construct ray directions
    rays = erp2xyz(uvrs)[..., None] # BNR31
    rot = poses[..., None, None, :3, :3] # B1133
    rays = tf.matmul(rot, rays)[..., 0] # BNR3
    lengths = tf.norm(rays, axis=-1, keepdim=True)
    dirs = rays / lengths
    
    # construct ray origins
    origins = poses[:, None, None, :3, 3] # B113
    origins = tf.repeat(origins, (1, N, R, 1)) # BNR3
    
    # return rays
    return tf.reshape(tf.cat([origins, dirs, lengths], axis=-1), (B, N * R, 7))
