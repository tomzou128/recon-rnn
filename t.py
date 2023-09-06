import torch
import numpy as np
import os
import cv2
import kornia
import matplotlib.pyplot as plt


def read_img(filepath):
    img = cv2.imread(filepath).astype(np.float32) / 255
    img = cv2.resize(img, (640, 480))
    return torch.tensor(img)


def read_gt_depth(filepath):
    # Read depth image and camera pose
    gt_depth = cv2.imread(filepath, -1).astype(np.float32)
    # gt_depth = cv2.resize(gt_depth, (64, 48), interpolation=cv2.INTER_NEAREST)
    gt_depth /= 1000.  # depth is saved in 16-bit PNG in millimeters
    return torch.tensor(gt_depth, dtype=torch.float32)


def read_cam_file(filepath, vid):
    intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_depth.txt'), delimiter=' ')[:3, :3]
    intrinsics = intrinsics.astype(np.float32)
    poses = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
    return torch.tensor(intrinsics, dtype=torch.float32), torch.tensor(poses, dtype=torch.float32)


def warp_frame_depth(
        image_src: torch.Tensor,
        depth_dst: torch.Tensor,
        src_trans_dst: torch.Tensor,
        camera_matrix: torch.Tensor,
        normalize_points: bool = False,
        sampling_mode='bilinear') -> (torch.Tensor, torch.Tensor):
    # TAKEN FROM KORNIA LIBRARY
    if not isinstance(image_src, torch.Tensor):
        raise TypeError(f"Input image_src type is not a torch.Tensor. Got {type(image_src)}.")

    if not len(image_src.shape) == 4:
        raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

    if not isinstance(depth_dst, torch.Tensor):
        raise TypeError(f"Input depht_dst type is not a torch.Tensor. Got {type(depth_dst)}.")

    if not len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1:
        raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

    if not isinstance(src_trans_dst, torch.Tensor):
        raise TypeError(f"Input src_trans_dst type is not a torch.Tensor. "
                        f"Got {type(src_trans_dst)}.")

    if not len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (3, 3):
        raise ValueError(f"Input src_trans_dst must have a shape (B, 3, 3). "
                         f"Got: {src_trans_dst.shape}.")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")
    # unproject source points to camera frame
    points_3d_dst: torch.Tensor = kornia.depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destination
    points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = kornia.transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3
    # points_3d_src[:, :, :, 2] = torch.relu(points_3d_src[:, :, :, 2])
    mask = points_3d_src[:, :, :, 2] > 0

    # project back to pixels
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src: torch.Tensor = kornia.project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]
    points_2d_src_norm: torch.Tensor = kornia.normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

    warp_feature = torch.nn.functional.grid_sample(image_src, points_2d_src_norm, align_corners=False,
                                                   mode=sampling_mode)

    mask_3 = mask.repeat(1, 3, 1, 1)
    warp_feature[~mask_3] = 0.0
    return warp_feature, mask


im_1, im_2 = 70, 73
img_1 = read_img(f"F:/D/ScanNetv1/scans/scene0000_00/color/{im_1}.jpg")
img_2 = read_img(f"F:/D/ScanNetv1/scans/scene0000_00/color/{im_2}.jpg")
depth_1 = read_gt_depth(f"F:/D/ScanNetv1/scans/scene0000_00/depth/{im_1}.png")
depth_2 = read_gt_depth(f"F:/D/ScanNetv1/scans/scene0000_00/depth/{im_2}.png")
in_1, ext_1 = read_cam_file("F:/D/ScanNetv1/scans/scene0000_00/", im_1)
in_2, ext_2 = read_cam_file("F:/D/ScanNetv1/scans/scene0000_00/", im_2)
# trans_2_to_1 = torch.mm(ext_1, torch.inverse(ext_2)).unsqueeze(0)
trans_2_to_1 = torch.mm(torch.inverse(ext_1), ext_2).unsqueeze(0)
# transformation = torch.eye(4).unsqueeze(0)

img_1_b = img_1.permute(2, 0, 1).unsqueeze(0)
img_2_b = img_2.permute(2, 0, 1).unsqueeze(0)
print("raw image comparison")
print(torch.norm(img_1_b - img_2_b))
# print(torch.norm(depth_1 - img_1_b))
# print(torch.norm(img_1_b - img_1_b))
depth_1_b = depth_1[None, None, ...]
depth_2_b = depth_2[None, None, ...]

wrap_img, mask = warp_frame_depth(img_1_b, depth_2_b, trans_2_to_1, torch.tensor(in_1).unsqueeze(0))

# msk = (depth_2_b > 0).repeat(1, 3, 1, 1)
# print(torch.norm(wrap_img[msk] - img_1_b[msk]))
# print(torch.norm(wrap_img[msk] - img_2_b[msk]))
print(wrap_img)
print(torch.norm(wrap_img - img_1_b))
print(torch.norm(wrap_img - img_2_b))
mask = mask.repeat(1, 3, 1, 1)
print(torch.norm(wrap_img[mask] - img_1_b[mask]))
print(torch.norm(wrap_img[mask] - img_2_b[mask]))

plt.imshow(img_1_b.squeeze(0).permute(1, 2, 0))
plt.title("img_1")
plt.show()
plt.imshow(wrap_img.squeeze(0).permute(1, 2, 0))
plt.title("wrap_img")
plt.show()
plt.imshow(img_2_b.squeeze(0).permute(1, 2, 0))
plt.title("img_2")
plt.show()
