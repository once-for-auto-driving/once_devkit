import json
import functools
import os.path as osp
from collections import defaultdict
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def split_info_loader_helper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        split_file_path = func(*args, **kwargs)
        if not osp.isfile(split_file_path):
            split_list = None
        else:
            split_list = set(map(lambda x: x.strip(), open(split_file_path).readlines()))
        return split_list
    return wrapper


class ONCE(object):
    """
    dataset structure:
    - data_root
        -ImageSets
            - train_split.txt
            - val_split.txt
            - test_split.txt
            - raw_split.txt
        - data
            - seq_id
                - cam01
                - cam03
                - ...
                -
    """
    camera_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
    camera_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']

    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.data_root = osp.join(self.dataset_root, 'data')
        self._collect_basic_infos()

    @property
    @split_info_loader_helper
    def train_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'train.txt')

    @property
    @split_info_loader_helper
    def val_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'val.txt')

    @property
    @split_info_loader_helper
    def test_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'test_set.txt')

    @property
    @split_info_loader_helper
    def raw_small_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_small.txt')

    @property
    @split_info_loader_helper
    def raw_medium_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_medium.txt')

    @property
    @split_info_loader_helper
    def raw_large_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_large.txt')

    def _find_split_name(self, seq_id):
        if seq_id in self.raw_small_split_list:
            return 'raw_small'
        elif seq_id in self.raw_medium_split_list:
            return 'raw_medium'
        elif seq_id in self.raw_large_split_list:
            return 'raw_large'
        if seq_id in self.train_split_list:
            return 'train'
        if seq_id in self.test_split_list:
            return 'test'
        if seq_id in self.val_split_list:
            return 'val'
        print("sequence id {} corresponding to no split".format(seq_id))
        raise NotImplementedError

    def _collect_basic_infos(self):
        self.train_info = defaultdict(dict)
        self.val_info = defaultdict(dict)
        self.test_info = defaultdict(dict)
        self.raw_small_info = defaultdict(dict)
        self.raw_medium_info = defaultdict(dict)
        self.raw_large_info = defaultdict(dict)

        for attr in ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']:
            if getattr(self, '{}_split_list'.format(attr)) is not None:
                split_list = getattr(self, '{}_split_list'.format(attr))
                info_dict = getattr(self, '{}_info'.format(attr))
                for seq in split_list:
                    anno_file_path = osp.join(self.data_root, seq, '{}.json'.format(seq))
                    if not osp.isfile(anno_file_path):
                        print("no annotation file for sequence {}".format(seq))
                        raise FileNotFoundError
                    anno_file = json.load(open(anno_file_path, 'r'))
                    frame_list = list()
                    for frame_anno in anno_file['frames']:
                        frame_list.append(frame_anno['frame_id'])
                        info_dict[seq][frame_anno['frame_id']] = {
                            'pose': frame_anno['pose'],
                        }
                        info_dict[seq][frame_anno['frame_id']]['calib'] = dict()
                        for cam_name in self.__class__.camera_names:
                            info_dict[seq][frame_anno['frame_id']]['calib'][cam_name] = {
                                'cam_to_velo': np.array(anno_file['calib'][cam_name]['cam_to_velo']),
                                'cam_intrinsic': np.array(anno_file['calib'][cam_name]['cam_intrinsic']),
                                'distortion': np.array(anno_file['calib'][cam_name]['distortion'])
                            }
                    info_dict[seq]['frame_list'] = sorted(frame_list)

    def get_frame_anno(self, seq_id, frame_id):
        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
        if 'anno' in frame_info:
            return frame_info['anno']
        return None

    def load_point_cloud(self, seq_id, frame_id):
        bin_path = osp.join(self.data_root, seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    def load_image(self, seq_id, frame_id, cam_name):
        cam_path = osp.join(self.data_root, seq_id, cam_name, '{}.jpg'.format(frame_id))
        img_buf = cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB)
        return img_buf

    def undistort_image(self, seq_id, frame_id):
        img_list = []
        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
        for cam_name in self.__class__.camera_names:
            img_buf = self.load_image(seq_id, frame_id, cam_name)
            cam_calib = frame_info['calib'][cam_name]
            h, w = img_buf.shape[:2]
            cv2.getOptimalNewCameraMatrix(cam_calib['cam_intrinsic'],
                                          cam_calib['distortion'],
                                          (w, h), alpha=0.0, newImgSize=(w, h))
            img_list.append(cv2.undistort(img_buf, cam_calib['cam_intrinsic'],
                                          cam_calib['distortion'],
                                          newCameraMatrix=cam_calib['cam_intrinsic']))
        return img_list

    def project_lidar_to_image(self, seq_id, frame_id):
        points = self.load_point_cloud(seq_id, frame_id)

        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
        points_img_dict = dict()
        img_list = self.undistort_image(seq_id, frame_id)
        for cam_no, cam_name in enumerate(self.__class__.camera_names):
            calib_info = frame_info['calib'][cam_name]
            cam_2_velo = calib_info['cam_to_velo']
            cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])
            point_xyz = points[:, :3]
            points_homo = np.hstack(
                [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            points_img = points_img / points_img[:, [2]]
            img_buf = img_list[cam_no]
            for point in points_img:
                try:
                    cv2.circle(img_buf, (int(point[0]), int(point[1])), 2, color=(0, 0, 255), thickness=-1)
                except:
                    print(int(point[0]), int(point[1]))
            points_img_dict[cam_name] = img_buf
        return points_img_dict

    def frame_concat(self, seq_id, frame_id, concat_cnt=0):
        """
        return new points coordinates according to pose info
        :param seq_id:
        :param frame_id:
        :return:
        """
        split_name = self._find_split_name(seq_id)

        seq_info = getattr(self, '{}_info'.format(split_name))[seq_id]
        start_idx = seq_info['frame_list'].index(frame_id)
        points_list = []
        translation_r = None
        for i in range(start_idx, start_idx + concat_cnt + 1):
            current_frame_id = seq_info['frame_list'][i]
            frame_info = seq_info[current_frame_id]
            transform_data = frame_info['pose']

            points = self.load_point_cloud(seq_id, current_frame_id)
            points_xyz = points[:, :3]

            rotation = Rotation.from_quat(transform_data[:4]).as_matrix()
            translation = np.array(transform_data[4:]).transpose()
            points_xyz = np.dot(points_xyz, rotation.T)
            points_xyz = points_xyz + translation
            if i == start_idx:
                translation_r = translation
            points_xyz = points_xyz - translation_r
            points_list.append(np.hstack([points_xyz, points[:, 3:]]))
        return points_list


if __name__ == '__main__':
    dataset = ONCE('./data')
    print(len(dataset.train_split_list))
    print(len(dataset.raw_small_split_list))
    # points_list = dataset.frame_concat('000000', '1616003650999', 3)
    # import pickle
    # pickle.dump(points_list, open('concat_sample.pkl', 'wb'))
    # img_dict = dataset.project_lidar_to_image('000000', '1616003650999')
    # cv2.imwrite('lidar_project.png', img_dict['cam01'])
