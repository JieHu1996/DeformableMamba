from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DMSDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("hide", "bone", "brick", "cardboard", "carpet","ceilingtile", "ceramic", 
                 "chalkboard", "clutter", "concrete", "cork", "engineeredstone", "fabric", 
                 "fire", "foliage", "food", "fur", "gemstone", "glass", "hair", "ice", 
                 "leather", "metal", "mirror", "paint", "paper", "photograph", "clearplastic", 
                 "plastic", "rubber", "sand", "skin", "sky", "snow", "soil", "stone", 
                 "polishedstone", "tile", "wallpaper", "water", "wax", "whiteboard", 
                 "wicker", "wood", "treewood", "asphalt"),
        palette=[[188, 188, 137],[0  , 188, 0  ],[188, 188, 0  ],[0  , 0  , 188],[188, 0  , 188],
                 [0  , 188, 188],[241, 241, 241],[0  , 137, 137],[225, 0  , 0  ],[137, 188, 0  ],
                 [225, 188, 0  ],[137, 0  , 188],[137, 188, 188],[225, 188, 188],[0  , 137, 0  ],
                 [188, 137, 0  ],[137, 225, 188],[188, 137, 188],[0  , 137, 188],[188, 225, 0  ],
                 [188, 225, 188],[137, 137, 0  ],[137, 225, 0  ],[225, 225, 0  ],[225, 137, 188],
                 [0  , 225, 0  ],[0  , 0  , 137],[188, 0  , 0  ],[0  , 188, 137],[188, 0  , 137],
                 [0  , 0  , 225],[225, 188, 137],[0  , 188, 225],[188, 188, 225],[225, 0  , 137],
                 [225, 225, 188],[137, 0  , 225],[137, 188, 225],[225, 188, 225],[0  , 137, 225],
                 [188, 137, 137],[188 ,188, 188],[188, 225, 137],[137, 0  , 0  ],[188, 137, 225],[137, 137, 137]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
