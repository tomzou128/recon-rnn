import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


# dataset_name = 'scannet_depth'
class ScanNetDatasetDepthRNN(Dataset):
    def __init__(self, cfg, transforms, scene=None, load_pred=True, load_gt=True):
        super(ScanNetDatasetDepthRNN, self).__init__()
        self.cfgg = cfg.TRAIN if cfg.MODE == 'train' else cfg.TEST

        self.datapath = self.cfgg.PATH
        self.tuplepath = self.cfgg.TUPLE_PATH
        self.mode = cfg.MODE
        self.n_views = self.cfgg.N_VIEWS
        self.transforms = transforms
        self.scene = scene

        assert self.mode in ['train', 'val', 'test']
        self.source_path = 'scans_test' if self.mode == 'test' else 'scans'

        self.epoch = None
        self.load_pred = load_pred
        self.load_gt = load_gt
        self.metas = self.build_list()

    def build_list(self):
        if self.scene is not None:
            scenes = [self.scene]
        else:
            # with open(os.path.join(self.tuplepath, self.mode + '.txt')) as f:
            with open(os.path.join(self.tuplepath, 'train.txt')) as f:
                scenes = f.read().splitlines()

        print(f'Found {len(scenes)} scenes')
        depth_ids_list = []
        for scene in scenes:
            depth_names = os.listdir(os.path.join(self.datapath, self.source_path, scene, 'sr_pred_depth'))
            depth_ids = [int(name.split("_")[0]) for name in depth_names]
            depth_ids.sort()
            r = len(depth_ids) % self.n_views
            s = np.random.randint(0, r+1)
            e = len(depth_ids) - (r - s)
            depth_ids = np.asarray(depth_ids[s:e]).reshape(-1, self.n_views)
            scene_list = [scene] * len(depth_ids)
            depth_ids = np.column_stack((scene_list, depth_ids))
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

    def read_img(self, filepath):
        img = Image.open(filepath)
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

        # imgs = []
        pred_depths = []
        gt_depths = []
        poses_list = []
        intrinsics_list = []

        for i, vid in enumerate(depth_ids):
            intrinsics, poses = self.read_cam_file(os.path.join(self.datapath, self.source_path, scene), vid)
            intrinsics_list.append(intrinsics)
            poses_list.append(poses)

            pred_depths.append(self.read_pred_depth(
                os.path.join(self.datapath, self.source_path, scene, 'sr_pred_depth', f'{vid}_pred_depth.npy')))

        items = {
            'scene': scene,
            # 'depth_ids': depth_ids,
            # 'epoch': [self.epoch],
            'pred_depths': pred_depths,
            'intrinsics': np.stack(intrinsics_list),
            'poses': np.stack(poses_list)
        }

        # items = {
        #     'imgs': imgs,
        #     'intrinsics': intrinsics,
        #     'poses': poses,
        #     'vol_origin': meta['vol_origin'],
        #     'scene': meta['scene'],
        #     'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
        #     'epoch': [self.epoch],
        #     'image_ids': meta['image_ids']
        # }

        if self.load_gt:
            for i, vid in enumerate(depth_ids):
                gt_depths.append(self.read_gt_depth(
                    os.path.join(self.datapath, self.source_path, scene, 'depth', '{}.png'.format(vid))))
            items['gt_depths'] = gt_depths

        if self.transforms is not None:
            items = self.transforms(items)

        return items
