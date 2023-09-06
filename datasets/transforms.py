""" Derived from [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Jiaming Sun and Yiming Xie. """

from PIL import Image, ImageOps
import cv2
import numpy as np
import torch


class Compose(object):
    """ Apply a list of transforms sequentially """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class ToTensor(object):
    """ Convert to torch tensors """

    def __call__(self, data):
        # data['imgs'] = torch.Tensor(np.stack(data['imgs']).transpose([0, 3, 1, 2]))
        data['intrinsics'] = torch.Tensor(data['intrinsics'])
        data['poses'] = torch.Tensor(data['poses'])
        # data['depth_ids'] = torch.Tensor(np.stack(data['depth_ids']))
        data['pred_depths'] = torch.Tensor(np.stack(data['pred_depths']))
        if 'gt_depths' in data.keys():
            data['gt_depths'] = torch.Tensor(np.stack(data['gt_depths']))
        # if 'tsdf_list_full' in data.keys():
        #     for i in range(len(data['tsdf_list_full'])):
        #         if not torch.is_tensor(data['tsdf_list_full'][i]):
        #             data['tsdf_list_full'][i] = torch.Tensor(data['tsdf_list_full'][i])
        return data


class IntrinsicsPoseToProjection(object):
    """ Convert intrinsics and extrinsics matrices to a single projection matrix """

    def __init__(self, n_views, stride=1):
        self.nviews = n_views
        self.stride = stride

    def rotate_view_to_align_xyplane(self, Tr_camera_to_world):
        # world space normal [0, 0, 1], camera space normal [0, -1, 0]
        z_c = np.dot(np.linalg.inv(Tr_camera_to_world), np.array([0, 0, 1, 0]))[: 3]
        axis = np.cross(z_c, np.array([0, -1, 0]))
        axis = axis / np.linalg.norm(axis)
        theta = np.arccos(-z_c[1] / (np.linalg.norm(z_c)))
        quat = transforms3d.quaternions.axangle2quat(axis, theta)
        rotation_matrix = transforms3d.quaternions.quat2mat(quat)
        return rotation_matrix

    def __call__(self, data):
        middle_pose = data['poses'][self.nviews // 2]
        rotation_matrix = self.rotate_view_to_align_xyplane(middle_pose)
        rotation_matrix4x4 = np.eye(4)
        rotation_matrix4x4[:3, :3] = rotation_matrix
        data['world_to_aligned_camera'] = torch.from_numpy(rotation_matrix4x4).float() @ middle_pose.inverse()

        proj_matrices = []
        proj_matrices_inv = []
        for intrinsics, poses in zip(data['intrinsics'], data['poses']):
            view_proj_matrics = []
            view_proj_matrics_inv = []
            for i in range(4):
                proj_mat = torch.inverse(poses.data.cpu())
                if i == 0:
                    # project to input images
                    scale_intrinsics = intrinsics
                else:
                    # project to feature maps
                    scale_intrinsics = intrinsics / self.stride / 2 ** (i - 1)
                    scale_intrinsics[-1, -1] = 1
                proj_mat[:3, :4] = scale_intrinsics @ proj_mat[:3, :4]
                view_proj_matrics.append(proj_mat)
                view_proj_matrics_inv.append(torch.inverse(proj_mat))
            proj_matrices.append(torch.stack(view_proj_matrics))
            proj_matrices_inv.append(torch.stack(view_proj_matrics_inv))

        data['proj_matrices'] = torch.stack(proj_matrices)
        data['proj_matrices_inv'] = torch.stack(proj_matrices_inv)

        return data


def pad_scannet(img, intrinsics):
    """ Scannet images are 1296x968 but 1296x972 is 4x3,
    so we pad vertically 4 pixels to make it 4x3 """

    w, h = img.size
    if w == 1296 and h == 968:
        img = ImageOps.expand(img, border=(0, 2))  # default zero padding
        intrinsics[1, 2] += 2
    return img, intrinsics


class ResizeImage(object):
    """ Resize everything to given size.
        Intrinsics are assumed to refer to image prior to resize.
        After resize everything (ex: depth) should have the same intrinsics. """

    def __init__(self, size, key='imgs'):
        self.size = size
        self.key = key

    def __call__(self, data):
        for i, im in enumerate(data[self.key]):
            im, intrinsics = pad_scannet(im, data['intrinsics'][i])
            w, h = im.size
            im = im.resize(self.size, Image.BILINEAR)
            intrinsics[0, :] /= (w / self.size[0])
            intrinsics[1, :] /= (h / self.size[1])

            data[self.key][i] = np.array(im, dtype=np.float32)
            data['intrinsics'][i] = intrinsics

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ResizeDepth(object):
    """ Resize depth map to given size.
        Intrinsics are assumed to refer to image prior to resize.
        After resize everything (ex: depth) should have the same intrinsics. """

    def __init__(self, size, key='gt_depths'):
        self.size = size
        self.key = key

    def __call__(self, data):
        for i, im in enumerate(data[self.key]):
            intrinsics = data['intrinsics'][i]
            h, w = im.shape
            im = cv2.resize(im, (self.size[0], self.size[1]), interpolation=cv2.INTER_NEAREST)
            intrinsics[0, :] /= (w / self.size[0])
            intrinsics[1, :] /= (h / self.size[1])

            data[self.key][i] = np.asarray(im, dtype=np.float32)
            data['intrinsics'][i] = intrinsics

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)