import cv2
import numpy as np
import tensorflow as tf
import random
import os
import imageio
from tqdm import tqdm
from run_nerf_helpers import *

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


def get_image(location):
    return None


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--data_location", type=str,
                        default='./data/lbx/sl2', help='location of foreground images and \
                        calibration data, can be a directory or s3 location')
    parser.add_argument("--bg_data_location", type=str,
                        default=None, help='directory where background-only images are stored')
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
    
    factor = args.scale_factor
    data_loc = args.data_location
    if 's3:' in data_loc:
        base_dir = './data/lbx/' + data_loc[data_loc.rindex('/') + 13:]
        os.makedirs(base_dir, exist_ok=True)
        
        os.makedirs(os.path.join(base_dir, 'calibration_data'), exist_ok=True)
        #TODO download calibration files from S3
    else:
        base_dir = data_loc
    
    # generate poses from calibration files
    
    intrinsics = np.load(os.path.join(base_dir, 'calibration_data', 'cam_mtx_list.npy'))
    extrinsics = np.load(os.path.join(base_dir, 'calibration_data', 'cam_extrinsics.npy'))
    locations = np.load(os.path.join(base_dir, 'calibration_data', 'cam_locations.npy'))
    distortions = np.load(os.path.join(base_dir, 'calibration_data', 'cam_dist_list.npy'))

    # Cosine and sine of 180 degrees
    c = -1 
    s = 0
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])
    
    poses_list = []
    
    for index in range(extrinsics.shape[0]):
        ext = extrinsics[index]
        loc = locations[index]

        # Figure out which "row" this camera is in: we have to do this because the matrix of
        # camera intrinsics has shape (27, 3, 3), while the other two arrays have shape (3240, ...) - 
        # this is because we only create one intrinsic matrix per "row" of the src cam locations
        row = index // 120
        intr = intrinsics[row]

        # Apply an additional rotation of 180-degrees about the x-axis (OpenCV -> OpenGL convention)
        #ext = np.matmul(ext, rot_x)
        ext = np.matmul(rot_x, ext)

        # Convert world-to-camera to camera-to-world
        ext = np.linalg.inv(ext)

        # Get rid of the last row, which is always(0, 0, 0, 1)
        ext = ext[:3, :]

        poses_list.append(ext)
    
    poses = np.array(poses_list)
    
    # sample from subset of evenly spaced camera poses
    indices_subset = select_camera_indices()
    
    num_test_imgs = args.num_test_imgs
    test_indices = indices_subset[np.linspace(0, indices_subset.size, num_test_imgs).astype(np.int32)]
    test_poses = []
    test_fxfycxcy = []
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    print('test indices: ', test_indices)
    
    num_v_imgs = args.num_v_imgs
    remaining_indices = np.array([x if x not in test_indices for x in indices_subset])
    v_indices = remaining_indices[np.linspace(0, remaining_indices.size, num_v_imgs).astype(np.int32)]
    v_poses = []
    v_fxfycxcy = []
    v_dir = os.path.join(base_dir, "validation")
    os.makedirs(os.path.join(v_dir, 'images'), exist_ok=True)
    print('validation indices: ', v_indices)
    
    num_training_images = indices_subset.size - num_v_imgs - num_test_imgs
    rays_per_img = np.ceil(num_rays / num_training_images).astype(np.int32)
    if rays_per_img > (sh[0] * sh[1]):
        rays_per_img = sh[0] * sh[1]
        
    num_rays_to_load = num_training_images * rays_per_img
    rays_od = np.zeros((num_rays_to_load, 2, 3), dtype=np.float32) # [ray_number, origin or direction, xyz]
    rays_rgb = np.zeros((num_rays_to_load, 2, 3), dtype=np.uint8) # [ray_number, foreground or background, rgb]
    
    # main processing loop
    
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

        # check if this image is a validation or test image
        if index in v_indices:
            save_path = os.join(v_dir, 'images', str(index).zfill(4) + '.jpg')
            imageio.imwrite(save_path, img.astype(np.uint8))
            v_poses.append(pose)
            v_fxfycxcy.append(fxfycxcy[index])
            continue
            
        elif index in test_indices:
            save_path = os.join(test_dir, 'images', str(index).zfill(4) + '.jpg')
            imageio.imwrite(save_path, img.astype(np.uint8))
            test_poses.append(pose)
            test_fxfycxcy.append(fxfycxcy[index])
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
    
    # truncate ray arrays to be exactly of size num_rays
    if rays_rgb.size > num_rays:
        rays_rgb = rays_rgb[:num_rays]
        rays_od = rays_od[:num_rays]
        
    # save test and validation data
    np.save(os.path.join(v_dir, 'poses'), np.array(v_poses))
    np.save(os.path.join(v_dir, 'fxfycxcy'), np.array(v_fxfycxcy))
    np.save(os.path.join(test_dir, 'poses'), np.array(test_poses))
    np.save(os.path.join(test_dir, 'fxfycxcy'), np.array(test_fxfycxcy))

if __name__ == '__main__':
    main(sys.argv[1:])