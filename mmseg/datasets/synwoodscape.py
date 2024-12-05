from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class SynWoodScapeDataset(BaseSegDataset):

    METAINFO = dict(
        classes=['building', 'fence', 'other', 
                 'pedestrian', 'pole', 'road line', 
                 'road', 'sidewalk', 'vegetation', 
                 'four-wheeler vehicle', 'wall', 
                 'traffic sign', 'sky', 'ground', 
                 'bridge', 'rail track', 
                 'guard rail', 'traffic light', 'water', 
                 'terrain', 'two-wheeler vehicle', 
                 'static', 'dynamic', 'ego-vehicle'],  # background is not included
        palette=[[ 70,  70,  70], [100,  40,  40], [ 55,  90,  80], 
                 [220,  20,  60], [153, 153, 153], [157, 234,  50], 
                 [128,  64, 128], [244,  35, 232], [107, 142,  35],
                 [  0,   0, 142], [102, 102, 156], [220, 220,   0],
                 [ 70, 130, 180], [ 81,   0,  81], [150, 100, 100],
                 [230, 150, 140], [180, 165, 180], [250, 170,  30],
                 [ 45,  60, 150], [145, 170, 100], [  0,   0, 230],
                 [110, 190, 160], [170, 120,  50], [255, 255, 255],])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)