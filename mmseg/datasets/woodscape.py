from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class WoodScapeDataset(BaseSegDataset):

    METAINFO = dict(
        classes=['road', 'lanemarks', 'curb', 
                 'person', 'rider', 'vehicles', 
                 'bicycle', 'motorcycle', 'traffic_sign'],  # background is not included
        palette=[[255,0,255], [255,0,0], [0,255,0], 
                 [0,0,255], [255,255,255], [255,255,0], 
                 [0,255,255], [128,128,255], [0,128,128]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)