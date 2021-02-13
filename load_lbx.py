import cv2
import numpy as np
import tensorflow as tf
import random
import os
import imageio
from tqdm import tqdm

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
def select_camera_indices(indices_path):
    
    indices = []
    # values were taken from mike's even spacing calculator script
    row_spacing = np.array([20, 9, 6, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    for i in range(len(row_samples)):
        if i == 0:
            offset = 0
        else:
            offset = (indices[-1] + (row_spacing[i - 1] // 2)) % row_spacing[i]
        indices += range(offset + i * 120, (i + 1) * 120, row_spacing[i])
        # print(''.join(['X' if x in indices else ' ' for x in range(i*120, (i+1)*120)]))
    
    indices_np = np.array(indices)
    np.save(indices_path, indices_np)
    return indices_np



def load_lbx_data(basedir, num_rays, factor=2, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, subset_size=None, bgimgdir=None):
    
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
    indices_path = os.path.join(basedir, 'camera_indices.npy')
    if os.path.exists(indices_path):
        indices_subset = np.load(indices_path)
    else:
        indices_subset = select_camera_indices(indices_path)
    
    # subset each of the parallel arrays
    poses = poses[:,:,indices_subset]
    bds = bds[:,indices_subset]
    fxfycxcy = fxfycxcy[indices_subset,:]
    
    dist_poses = dist_poses[indices_subset]
    intr_poses = intr_poses[indices_subset]
    
    imgfiles = [imgfiles[i] for i in indices_subset]
    bgimgfiles = [bgimgfiles[i] for i in indices_subset]
    
    if not load_imgs:
        return poses, bds
    
    ray_o_d = np.zeros(
    imgs = np.zeros((sh[0], sh[1], 3, indices_subset.size), dtype=np.uint8)
    bgs = np.zeros((sh[0], sh[1], 3, indices_subset.size), dtype=np.uint8)
    
    for i in tqdm(range(subset_size)):
        # image resize expression taken from https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
        img = imageio.imread(imgfiles[i])[...,:3]
        img = cv2.undistort(img, intr_poses[i], dist_poses[i])
        imgs[...,i] = img.reshape(sh[0], factor, sh[1], factor, 3).mean(3).mean(1).astype(np.uint8)
        
        bgimg = imageio.imread(bgimgfiles[i])[...,:3]
        bgimg = cv2.undistort(bgimg, intr_poses[i], dist_poses[i])
        bgs[...,i] = bgimg.reshape(sh[0], factor, sh[1], factor, 3).mean(3).mean(1).astype(np.uint8)
        
#     imgs = imgs = [(imread(f)[...,:3]).astype(np.uint8) for f in imgfiles]
#     imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs, fxfycxcy, bgs
###
    

    poses, bds, imgs, fxfycxcy, bgs = _load_data(basedir, factor=factor, subset_size=subset_size, bgimgdir=bgimgdir) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0)
    bgs = np.moveaxis(bgs, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, imgs.shape, bgs.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    poses = poses.astype(np.float32)

    return imgs, poses, bds, render_poses, i_test, fxfycxcy, bgs