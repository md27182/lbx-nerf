import os
import sys
import cv2
import time
import random
import json
import tempfile
from functools import partial
import multiprocessing as mp

import numpy as np
import imageio
from tqdm import tqdm
import shutil
import boto3


# define Amazon S3 related constants
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
s3_client = boto3.client("s3", 
                      aws_access_key_id = AWS_ACCESS_KEY_ID, 
                      aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
bucket = "lbxlabs.scandata"

# this function makes a new s3_client for each process, read more here: https://stackoverflow.com/questions/48091874/downloading-multiple-s3-objects-in-parallel-in-python
def init_s3_client():
    global s3_client
    s3_client = boto3.client("s3", 
                      aws_access_key_id = AWS_ACCESS_KEY_ID, 
                      aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
    
# this function also adds 's3:' to the begining of every file name returned
def s3_list_dir(path, suffix):
    ret_list = []
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=(path))
    for page in pages:
        for obj in page['Contents']:
            ret_list.append(obj['Key'])
            
    ret_list = [('s3:' + x) for x in ret_list if x.endswith(suffix)]
    ret_list.sort()
    return ret_list
    
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
    
def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w
    
def normalize(x):
    return x / np.linalg.norm(x)
    
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m
    
def spherify_poses(poses, bds):

    # transpose bounds to make them match bmild format
    bds = np.transpose(bds)

    # add last col to poses to make them match bmild format
    hwf = np.zeros((3240, 3, 1))
    poses = np.concatenate((poses, hwf), axis=2)
    
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

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1))) # radius?
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc # rad = 1?
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    # generates new render path
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
    
    # retranslate back to lbx format
    bds = np.transpose(bds)
    poses = poses[:, :, :4]

    return poses_reset, bds


def get_rays_np(H, W, fxfycxcy, c2w):
    """Get ray origins, directions from a pinhole camera."""
    fx, fy, cx, cy = fxfycxcy
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cx)/fx, -(j-cy)/fy, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


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
    
    return np.array(indices)


def pose_from_ext(extrinsic):
    
    # Cosine and sine of 180 degrees
    c = -1 
    s = 0
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

    # Apply an additional rotation of 180-degrees about the x-axis (OpenCV -> OpenGL convention)
    temp = np.matmul(rot_x, extrinsic)

    # Convert world-to-camera to camera-to-world
    temp = np.linalg.inv(temp)

    # Get rid of the last row, which is always(0, 0, 0, 1)
    temp = temp[:3, :]
    
    return temp


def get_image(img_location):
    if 's3:' in img_location:
        img_location = img_location[3:]
        #S3 direct-to-RAM download
        file = s3_client.get_object(Bucket=bucket, Key=img_location)['Body'].read()
        return imageio.imread(file)
    else:
        return imageio.imread(img_location)
    
    
def process_one_image(rays_per_img, sh, factor, imgfiles, bgimgfiles, fxfycxcy, intrinsics, distortions, poses, v_dir, test_dir, v_indices, test_indices, index):

        row = index // 120
        
        # image resize expression taken from https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
        # should we use a gaussian blur instead of resizing?
        img = get_image(imgfiles[index])
        img = cv2.undistort(img, intrinsics[row], distortions[row])
        img = img.reshape(sh[0], factor, sh[1], factor, 3).mean(3).mean(1)
        
        bgimg = get_image(bgimgfiles[index])
        bgimg = cv2.undistort(bgimg, intrinsics[row], distortions[row])
        bgimg = bgimg.reshape(sh[0], factor, sh[1], factor, 3).mean(3).mean(1)
        
        #pose = pose_from_ext(extrinsics[index])
        pose = poses[index]

        # check if this image is a validation or test image
        if index in v_indices:
            save_path = os.path.join(v_dir, 'images', str(index).zfill(4) + '.jpg')
            imageio.imwrite(save_path, img.astype(np.uint8))
            return 'validation', pose, fxfycxcy[row]
            
        elif index in test_indices:
            save_path = os.path.join(test_dir, 'images', str(index).zfill(4) + '.jpg')
            imageio.imwrite(save_path, img.astype(np.uint8))
            return 'test', pose, fxfycxcy[row]
        
        h_img, w_img = img.shape[:2]
        h_bin, w_bin = 200, 200
        
        val_thresh = 10
        count_thresh = 0.05 * (3 * h_bin * w_bin)

        # count the number of r, g, and b diffs in each bin that exceed val_thresh
        diff = img - bgimg
        matte = np.abs(diff) > val_thresh
        matte = matte.reshape(h_img // h_bin, h_bin, w_img // w_bin, w_bin, 3).sum(3).sum(1).sum(-1)
        
        # find the bins for which more than 5% of the pixels exceed val_thresh, TODO make sure not all bins are False
        matte = matte > count_thresh
        
        # scale the matte back up to h_img, w_img
        matte = np.repeat(np.repeat(matte, h_bin, axis=0), w_bin, axis=1)
        
        # get rays from image
        rays_o, rays_d = get_rays_np(h_img, w_img, fxfycxcy[row], pose)
        
        coords = np.stack(np.meshgrid(np.arange(h_img), np.arange(w_img), indexing='ij'), -1)
        good_coords = coords[matte]
        good_coords = np.reshape(good_coords, [-1, 2])
        
        select_inds = np.random.choice(good_coords.shape[0], size=[rays_per_img], replace=False)
        select_inds = np.transpose(good_coords[select_inds])
        
        rays_o = rays_o[select_inds[0], select_inds[1], :]
        rays_d = rays_d[select_inds[0], select_inds[1], :]
        
        rays_rgb_fg = img.astype(np.uint8)[select_inds[0], select_inds[1], :]
        rays_rgb_bg = bgimg.astype(np.uint8)[select_inds[0], select_inds[1], :]
        
        return 'training', rays_o, rays_d, rays_rgb_fg, rays_rgb_bg


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--data_location", type=str,
                        default='./data/lbx/sl2', help='location of foreground images and \
                        calibration data, can be a directory (e.g. ./data/lbx/sl2) or s3 location \
                        (e.g. s3:210203220030_helicopter)')
    parser.add_argument("--bg_data_location", type=str,
                        default=None, help='location of background-only images, \
                        can be a directory (e.g. ./data/lbx/bg001) or s3 location \
                        (e.g. s3:bg001)')
    parser.add_argument("--scale_factor", type=int,
                        default=2, help='new training images will be 4000/factor by 6000/factor')
    parser.add_argument("--num_v_imgs", type=int,
                        default=10, help='number of images in the validation set, these images wont \
                        be used for training but may be used for hyperparameter tuning')
    parser.add_argument("--num_test_imgs", type=int,
                        default=3, help='number of images in the test set, these images are only used \
                        for final analysis')
    parser.add_argument("--num_rays", type=int,
                        default=1000000, help='target number or rays in processed dataset, final number \
                        may differ from this if there are not enough rays')
    
    return parser


def main(_args):
    
    parser = config_parser()
    args = parser.parse_args(_args)
    
    num_rays = args.num_rays
    factor = args.scale_factor
    data_loc = args.data_location
    bg_data_loc = args.bg_data_location
    
    if 's3:' in data_loc:
        s3_loc = data_loc[3:]
        base_dir = './data/lbx/' + s3_loc
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'calibration_data'), exist_ok=True)
        
        # download calibration files from S3
        calibration_file_names = ['cam_mtx_list.npy', 'cam_extrinsics.npy', 'cam_dist_list.npy', 'cam_locations.npy']
        for f in calibration_file_names:
            object_name = s3_loc + '/calibration_data/' + f
            dest_path = os.path.join(base_dir, 'calibration_data', f)
            s3_client.download_file(bucket, object_name, dest_path)
        
        # create imgfiles by checking s3
        imgfiles = s3_list_dir(s3_loc + '/images', 'jpg')
        
    else:
        base_dir = data_loc
        imgdir = os.path.join(base_dir, 'images')
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) 
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        
    if 's3:' in bg_data_loc:
        s3_loc = bg_data_loc[3:]
        bgimgfiles = s3_list_dir(s3_loc + '/images', 'jpg')
    else:
        bgimgdir = os.path.join(bg_data_loc, 'images')
        bgimgfiles = [os.path.join(bgimgdir, f) for f in sorted(os.listdir(bgimgdir)) 
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        
    # load an image to find it's shape
    img0 = get_image(imgfiles[0])
    sh = ((1. / factor) * np.array(img0.shape)).astype(np.int32)
    
    # load calibration data
    intrinsics = np.load(os.path.join(base_dir, 'calibration_data', 'cam_mtx_list.npy'))
    extrinsics = np.load(os.path.join(base_dir, 'calibration_data', 'cam_extrinsics.npy'))
    distortions = np.load(os.path.join(base_dir, 'calibration_data', 'cam_dist_list.npy'))
    locations = np.load(os.path.join(base_dir, 'calibration_data', 'cam_locations.npy'))
    fxfycxcy = np.zeros((intrinsics.shape[0], 4))
    for row in range(intrinsics.shape[0]):
        fxfycxcy[row] = np.array([intrinsics[row, 0, 0], intrinsics[row, 1, 1], 
                                     intrinsics[row, 0, 2], intrinsics[row, 1, 2]]) 
    fxfycxcy = fxfycxcy * (1./factor)

    # create poses array
    poses = np.zeros((extrinsics.shape[0], 3, 4))
    for i in range(extrinsics.shape[0]):
        poses[i] = pose_from_ext(extrinsics[i])

    # generate bounds, not needed for NSVF
    dists = np.linalg.norm(locations, axis=1)
    bds = np.stack((dists - 24, 2.5 * dists))

    # recenter and spherify
    # poses = recenter_poses(poses) forgot to modify recenter function so just taking it out for now
    poses, bds = spherify_poses(poses, bds)
    
    # sample from subset of evenly spaced camera poses
    indices_subset = select_camera_indices()
    
    num_test_imgs = args.num_test_imgs
    test_indices = indices_subset[np.linspace(0, indices_subset.size - 1, num_test_imgs).astype(np.int32)]
    test_poses = []
    test_fxfycxcy = []
    test_dir = os.path.join(base_dir, "test")
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    print('test indices: ', test_indices)
    
    num_v_imgs = args.num_v_imgs
    remaining_indices = np.array([x for x in indices_subset if x not in test_indices])
    v_indices = remaining_indices[np.linspace(0, remaining_indices.size - 1, num_v_imgs).astype(np.int32)]
    v_poses = []
    v_fxfycxcy = []
    v_dir = os.path.join(base_dir, "validation")
    if os.path.isdir(v_dir):
        shutil.rmtree(v_dir)
    os.makedirs(os.path.join(v_dir, 'images'), exist_ok=True)
    print('validation indices: ', v_indices)
    
    num_training_images = indices_subset.size - num_v_imgs - num_test_imgs
    rays_per_img = np.ceil(num_rays / num_training_images)
    if rays_per_img > (sh[0] * sh[1]):
        rays_per_img = sh[0] * sh[1]
    rays_per_img = int(rays_per_img)
        
    num_rays_to_load = num_training_images * rays_per_img
    rays_od = np.zeros((num_rays_to_load, 2, 3), dtype=np.float32) # [ray_number, origin or direction, xyz]
    rays_rgb = np.zeros((num_rays_to_load, 2, 3), dtype=np.uint8) # [ray_number, foreground or background, rgb]
    
    # main processing loop
    num_cpus = mp.cpu_count()
    p_fn = partial(process_one_image, rays_per_img, sh, factor, imgfiles, bgimgfiles, fxfycxcy, intrinsics, distortions, poses, v_dir, test_dir, v_indices, test_indices)
    t_imgs_counter = 0
    
    with mp.Pool(num_cpus, init_s3_client()) as p:
        for ret in tqdm(p.imap_unordered(p_fn, indices_subset), total=indices_subset.size):
            
            if ret[0] == 'training':
                ray_numbers = range(t_imgs_counter * rays_per_img, (t_imgs_counter + 1) * rays_per_img)

                rays_od[ray_numbers, 0, :] = ret[1]
                rays_od[ray_numbers, 1, :] = ret[2]

                rays_rgb[ray_numbers, 0, :] = ret[3]
                rays_rgb[ray_numbers, 1, :] = ret[4]

                t_imgs_counter += 1
                
            elif ret[0] == 'validation':
                v_poses.append(ret[1])
                v_fxfycxcy.append(ret[2])
                
            elif ret[0] == 'test':
                test_poses.append(ret[1])
                test_fxfycxcy.append(ret[2])
    
    # truncate ray arrays to be exactly of size num_rays
    if rays_rgb.size > num_rays:
        rays_rgb = rays_rgb[:num_rays]
        rays_od = rays_od[:num_rays]
        
    # shuffle rays
    print('shuffling rays')
    shuffler = np.random.permutation(rays_rgb.shape[0])
    rays_rgb = rays_rgb[shuffler]
    rays_od = rays_od[shuffler]
    print('done')
    
    # save rays locally
    train_dir = os.path.join(base_dir, 'training')
    os.makedirs(train_dir, exist_ok=True)
    np.savez(os.path.join(train_dir, 'preprocessed.npz'), rays_od, rays_rgb)
        
    # save test and validation data
    np.save(os.path.join(v_dir, 'poses'), np.array(v_poses))
    np.save(os.path.join(v_dir, 'fxfycxcy'), np.array(v_fxfycxcy))
    np.save(os.path.join(test_dir, 'poses'), np.array(test_poses))
    np.save(os.path.join(test_dir, 'fxfycxcy'), np.array(test_fxfycxcy))

    # save bounds, not needed for NSVF
    np.save(os.path.join(base_dir, 'bds.npy'), bds)

    #TODO upload preprocessed dataset to s3

if __name__ == '__main__':
    main(sys.argv[1:])
