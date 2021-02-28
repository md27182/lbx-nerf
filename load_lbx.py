import cv2
import numpy as np
import tensorflow as tf
import random
import os
import imageio
from tqdm import tqdm
from run_nerf_helpers import *

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds


# selects 2134 evenly spaced poses out of the full 3240
def select_camera_indices():
    
    indices = []
    # values were taken from mike's even spacing calculator script
    row_spacing = np.array([20, 9, 6, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    for i in range(len(row_spacing)):
        if i == 0:
            offset = 0
        else:
            offset = (indices[-1] + (row_spacing[i - 1] // 2)) % row_spacing[i]
        indices += range(offset + i * 120, (i + 1) * 120, row_spacing[i])
        # print(''.join(['X' if x in indices else ' ' for x in range(i*120, (i+1)*120)]))
    
    indices_np = np.array(indices)
    # np.save(indices_path, indices_np)
    return indices_np

#def _load_data(basedir, bgimgdir, num_rays, holdout, factor):
    


def load_lbx_data(basedir, bgimgdir, num_rays, holdout, factor, recenter=True, spherify=True, path_zflat=False):
        
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    imgdir = os.path.join(basedir, 'images')
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    bgimgfiles = [os.path.join(bgimgdir, f) for f in sorted(os.listdir(bgimgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    # overwrite h,w,f in all poses
    sh = np.array(imageio.imread(imgfiles[0]).shape)
    sh = (sh * (1. / factor)).astype(np.int32)  # scale H, W by factor
    poses[:2, 4, :] = sh[:2].reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor # scale focal length by factor
    
    intrinsics = np.load(basedir + '/calibration_data/cam_mtx_list.npy')
    intr_poses = np.zeros((3240,) + intrinsics.shape[1:]) # new intrinsics array parallel to the poses array
    fxfycxcy = np.zeros((3240, 4))
    
    distortions = np.load(basedir + '/calibration_data/cam_dist_list.npy')
    dist_poses = np.zeros((3240,) + distortions.shape[1:]) # new distortions array parallel to the poses array
    
    for i in range(3240):
        row = i // 120
        fxfycxcy[i,:] = np.array([intrinsics[row, 0, 0], intrinsics[row, 1, 1], intrinsics[row, 0, 2], intrinsics[row, 1, 2]])
        dist_poses[i] = distortions[row]
        intr_poses[i] = intrinsics[row]
    
    fxfycxcy = fxfycxcy * (1./factor)
    
    
    # sample from subset of evenly spaced camera poses
    indices_subset = select_camera_indices()
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds) 
        
    poses = poses.astype(np.float32)
    render_poses = np.array(render_poses).astype(np.float32)
    
    num_val_imgs = np.ceil(indices_subset.size / holdout).astype(np.int32) # total number of validation images
    val_imgs = np.zeros((num_val_imgs, sh[0], sh[1], 3), dtype=np.uint8)
    val_indices = np.zeros(num_val_imgs).astype(np.int32)
    # val_poses = np.zeros(((num_val_imgs,) + poses.shape[1:]))
    
    num_training_images = indices_subset.size - num_val_imgs
    rays_per_img = np.ceil(num_rays / num_training_images).astype(np.int32)
    if rays_per_img > (sh[0] * sh[1]):
        rays_per_img = sh[0] * sh[1]
        
    num_rays_to_load = num_training_images * rays_per_img
    rays_od = np.zeros((num_rays_to_load, 2, 3), dtype=np.float32) # [ray_number, origin or direction, xyz]
    rays_rgb = np.zeros((num_rays_to_load, 2, 3), dtype=np.uint8) # [ray_number, foreground or background, rgb]
    
    t_imgs_counter = 0
    for i, index in tqdm(enumerate(indices_subset)):
        
        # image resize expression taken from https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
        img = imageio.imread(imgfiles[index])[...,:3]
        img = cv2.undistort(img, intr_poses[index], dist_poses[index])
        img = img.reshape(sh[0], factor, sh[1], factor, 3).mean(3).mean(1)
        
        bgimg = imageio.imread(bgimgfiles[index])[...,:3]
        bgimg = cv2.undistort(bgimg, intr_poses[index], dist_poses[index])
        bgimg = bgimg.reshape(sh[0], factor, sh[1], factor, 3).mean(3).mean(1)
        
        pose = poses[index, :3, :4]

        # check if this image should be a validation image
        if i % holdout == 0:
            val_imgs[i // holdout] = img.astype(np.uint8)
            val_indices[i // holdout] = index
            continue
        
        h_img, w_img = img.shape[:2]
        h_bin, w_bin = 200, 200
        
        val_thresh = 10
        count_thresh = 0.05 * (3 * h_bin * w_bin)

        # count the number of r, g, and b diffs in each bin that exceed val_thresh
        diff = img - bgimg
        matte = np.abs(diff) > val_thresh
        matte = matte.reshape(h_img // h_bin, h_bin, w_img // w_bin, w_bin, 3).sum(3).sum(1).sum(-1)
        # find the bins for which more than 5% of the pixels exceed val_thresh
        matte = matte > count_thresh

        # scale the matte back up to h_img, w_img
        matte = np.repeat(np.repeat(matte, h_bin, axis=0), w_bin, axis=1)
        
        
        # get rays from image
        rays_o, rays_d = get_rays(h_img, w_img, fxfycxcy[index], pose)
        
        ray_numbers = range(t_imgs_counter * rays_per_img, (t_imgs_counter + 1) * rays_per_img)

        coords = tf.stack(tf.meshgrid(tf.range(h_img), tf.range(w_img), indexing='ij'), -1)
        good_coords = tf.boolean_mask(coords, matte)
        good_coords = tf.reshape(good_coords, [-1, 2])

        select_inds = np.random.choice(good_coords.shape[0], size=[rays_per_img], replace=False)
        select_inds = tf.gather_nd(good_coords, select_inds[:, tf.newaxis])
        
        rays_od[ray_numbers, 0, :] = tf.gather_nd(rays_o, select_inds)
        rays_od[ray_numbers, 1, :] = tf.gather_nd(rays_d, select_inds)

        rays_rgb[ray_numbers, 0, :] = tf.gather_nd(img.astype(np.uint8), select_inds)
        rays_rgb[ray_numbers, 1, :] = tf.gather_nd(bgimg.astype(np.uint8), select_inds)
        
        t_imgs_counter += 1

    # subset the poses, bounds, and intrinsic values to only include the validation set
    val_poses = poses[val_indices,:,:]
    val_bds = bds[val_indices,:]
    val_fxfycxcy = fxfycxcy[val_indices,:]
    
    print('Loaded image data')
    print('Loaded', basedir, bds.min(), bds.max())

    print('Val data:')
    print(val_poses.shape, val_bds.shape, val_fxfycxcy.shape)
    

    return rays_od, rays_rgb, val_imgs, val_poses, val_fxfycxcy, render_poses, bds