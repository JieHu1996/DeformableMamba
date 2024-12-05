from .dms import DMSDataset
from .ade import ADE20KDataset
from .coco_stuff import COCOStuffDataset
from .coco_stuff_164k import COCOStuff164kDataset
from .s2d3d import S2D3DDataset
from .mp3d import MP3DDataset
from .synpass import SynPASSDataset
from .woodscape import WoodScapeDataset
from .synwoodscape import SynWoodScapeDataset
from .basesegdataset import BaseSegDataset
from .transforms import *


__all__ = ['BaseSegDataset', 'ADE20KDataset', 'COCOStuffDataset',
           'COCOStuff164kDataset', 'SynWoodScapeDataset',
           'DMSDataset', 'S2D3DDataset', 'MP3DDataset', 
           'WoodScapeDataset', 'SynPASSDataset']