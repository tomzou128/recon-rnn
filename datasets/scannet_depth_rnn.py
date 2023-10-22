import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from loguru import logger


# dataset_name = 'scannet_depth'
class ScanNetDatasetDepthRNN(Dataset):
    def __init__(self, cfg, transforms, scene=None):
        super(ScanNetDatasetDepthRNN, self).__init__()
        self.cfg = cfg.DATA

        self.data_path = self.cfg.PATH
        self.mode = cfg.MODE
        self.n_views = self.cfg.N_VIEWS
        self.transforms = transforms
        self.scene = scene

        assert self.mode in ['train', 'val', 'test']
        self.source_path = 'scans_test' if self.mode == 'test' else 'scans'

        # if self.cfg.TUPLE_FILE:
        #     self.tuple_file = self.cfg.TUPLE_FILE
        # else:
        #     self.tuple_file =

        if self.cfg.TUPLE_FILE:
            self.tuple_path = os.path.join(self.cfg.TUPLE_PATH, self.cfg.TUPLE_FILE)
        else:
            self.tuple_path = os.path.join(self.cfg.TUPLE_PATH, self.mode + '_views_3510.txt')

        # self.metas = self.build_list_from_file()
        # self.metas = self.build_list_from_folder()
        self.metas = self.build_list_from_folder_six()

    def build_list_from_folder(self):
        if self.scene is not None:
            scenes = [self.scene]
        else:
            # with open(os.path.join(self.tuple_path, self.mode + '.txt')) as f:
            # with open(os.path.join(self.tuple_path, 'train.txt')) as f:
            with open("data_splits/ScanNetv1/test.txt") as f:
                scenes = f.read().splitlines()

        print(f'Found {len(scenes)} scenes')
        depth_ids_list = []
        for scene in scenes:
            depth_names = os.listdir(os.path.join(self.data_path, self.source_path, scene, 'sr_pred_depth'))
            depth_ids = [int(name.split("_")[0]) for name in depth_names]
            depth_ids.sort()

            if self.mode == 'train':
                r = len(depth_ids) % self.n_views
                s = np.random.randint(0, r + 1)
                e = len(depth_ids) - (r - s)
            else:
                s, e = 0, len(depth_ids) - (len(depth_ids) % self.n_views)

            depth_ids = np.asarray(depth_ids[s:e]).reshape(-1, self.n_views)
            scene_list = [scene] * len(depth_ids)
            depth_ids = np.column_stack((scene_list, depth_ids))
            depth_ids_list.extend(depth_ids)
        print(f'Found {len(depth_ids_list)} samples')
        return depth_ids_list

    def build_list_from_file(self):
        logger.info(f'open {self.tuple_path}')
        with open(self.tuple_path) as f:
            depth_tuples = f.read().splitlines()

        # depth_tuples = depth_tuples[500:1140]

        if self.scene is not None:
            depth_tuples = [self.scene in depth_tuple for depth_tuple in depth_tuples]

        logger.info(f'Found {len(depth_tuples)} samples')
        depth_tuples = [depth_tuple.split(' ') for depth_tuple in depth_tuples]
        return depth_tuples

    def build_list_from_folder_six(self):
        with open("data_splits/ScanNetv1/test.txt") as f:
        # with open("data_splits/ScanNetv1/val.txt") as f:
            scenes = f.read().splitlines()

        print(f'Found {len(scenes)} scenes')
        depth_ids_list = []

        for scene in scenes:
            depth_names = os.listdir(os.path.join(self.data_path, self.source_path, scene, 'sr_pred_depth'))
            depth_ids = [int(name.split("_")[0]) for name in depth_names]
            depth_ids.sort()
            tmp_list = []

            for i in range(len(depth_ids) - 7):
                tmp_list.append(depth_ids[i:i + 8])

            scene_list = [scene] * len(tmp_list)
            depth_ids = np.column_stack((scene_list, tmp_list))
            depth_ids_list.extend(depth_ids)

        print(f'Found {len(depth_ids_list)} samples')
        return depth_ids_list

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_depth.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        poses = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, poses

    def read_img(self, filepath, size=None):
        img = Image.open(filepath)
        if size is not None:
            img = img.resize(size)
        return img

    def read_pred_depth(self, filepath):
        pred_depth = np.load(filepath).astype(np.float32)
        pred_depth /= 1000.
        return pred_depth

    def read_gt_depth(self, filepath):
        # Read depth image and camera pose
        gt_depth = cv2.imread(filepath, -1).astype(np.float32)
        gt_depth /= 1000.  # depth is saved in 16-bit PNG in millimeters
        return gt_depth

    def __getitem__(self, idx):
        scene, *depth_ids = self.metas[idx]

        imgs = []
        pred_depths = []
        gt_depths = []
        poses_list = []
        intrinsics_list = []

        for i, vid in enumerate(depth_ids):
            intrinsics, poses = self.read_cam_file(os.path.join(self.data_path, self.source_path, scene), vid)
            intrinsics_list.append(intrinsics)
            poses_list.append(poses)

            pred_depths.append(self.read_pred_depth(
                os.path.join(self.data_path, self.source_path, scene, 'sr_pred_depth', f'{vid}_pred_depth.npy')))
            gt_depths.append(self.read_gt_depth(
                os.path.join(self.data_path, self.source_path, scene, 'depth', '{}.png'.format(vid))))

        items = {
            'scene': scene,
            'pred_depths': pred_depths,
            'gt_depths': gt_depths,
            'intrinsics': intrinsics_list,
            'poses': poses_list
        }

        if self.cfg.LOAD_IMG:
            imgs = []
            for i, vid in enumerate(depth_ids):
                imgs.append(self.read_img(
                    os.path.join(self.data_path, self.source_path, scene, 'color', f'{vid}.jpg'),
                    size=pred_depths[i].shape[::-1]))
            items['imgs'] = imgs

        if self.transforms is not None:
            items = self.transforms(items)

        return items
