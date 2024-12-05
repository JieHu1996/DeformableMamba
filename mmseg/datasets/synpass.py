# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List
import mmengine
import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SynPASSDataset(BaseSegDataset):
    """SynPASS dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_trainID.png' for synPASS dataset.
    """
    METAINFO = dict(
        classes=('Building','Fence','Other','Pedestrian','Pole',
                 'RoadLine','Road','SideWalk','Vegetation','Vehicles',
                 'Wall','TrafficSign','Sky','Ground','Bridge',
                 'RailTrack','GroundRail','TrafficLight','Static','Dynamic',
                 'Water','Terrain'),
        palette=[])

    def __init__(self,
                 split,
                 img_suffix='.jpg',
                 seg_map_suffix='_trainID.png',
                 **kwargs) -> None:
        self.split = split
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        for weather in ['cloud', 'fog', 'rain', 'sun']:
            new_img_dir = osp.join(img_dir, weather, self.split)
            new_ann_dir = osp.join(ann_dir, weather, self.split)
            if not osp.isdir(self.ann_file) and self.ann_file:
                assert osp.isfile(self.ann_file), \
                    f'Failed to load `ann_file` {self.ann_file}'
                lines = mmengine.list_from_file(
                    self.ann_file, backend_args=self.backend_args)
                for line in lines:
                    img_name = line.strip()
                    data_info = dict(
                        img_path=osp.join(new_img_dir, img_name + self.img_suffix))
                    if new_ann_dir is not None:
                        seg_map = img_name + self.seg_map_suffix
                        data_info['seg_map_path'] = osp.join(new_ann_dir, seg_map)
                    data_info['label_map'] = self.label_map
                    data_info['reduce_zero_label'] = self.reduce_zero_label
                    data_info['seg_fields'] = []
                    data_list.append(data_info)
            else:
                _suffix_len = len(self.img_suffix)
                for img in fileio.list_dir_or_file(
                        dir_path=new_img_dir,
                        list_dir=False,
                        suffix=self.img_suffix,
                        recursive=True,
                        backend_args=self.backend_args):
                    data_info = dict(img_path=osp.join(new_img_dir, img))
                    if new_ann_dir is not None:
                        seg_map = img[:-_suffix_len] + self.seg_map_suffix
                        data_info['seg_map_path'] = osp.join(new_ann_dir, seg_map)
                    data_info['label_map'] = self.label_map
                    data_info['reduce_zero_label'] = self.reduce_zero_label
                    data_info['seg_fields'] = []
                    data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
