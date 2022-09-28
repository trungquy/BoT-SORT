import os
import torch
import torch.nn as nn
from tracker.bot_sort import BoTSORT


VERSION = 'v0.1.0'


def _default_config():
    class DotDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
        def copy(self):
            return DotDict(super(DotDict, self).copy())

    return DotDict(
        device = 'gpu',
        fp16 = True,
        fuse = True,
        trt = False,
        track_high_thresh = 0.6,
        track_low_thresh = 0.1,
        new_track_thresh = 0.7,
        track_buffer = 30,
        match_thresh = 0.8,
        aspect_ratio_thresh = 1.6,
        min_box_area = 10,
        fuse_score = True,
        fast_reid_config = None,
        fast_reid_weights = None,
        proximity_thresh = 0.5,
        appearance_thresh = 0.25,
        mot20 = False,
        name = None,
        ablation = None,
    )


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, detections, image):
        return self.model.update(detections, image)


def botsort(fps: float, with_reid: bool, cmc_method: str = 'orb', cmc_downscale: int = 4, mot_year: int = 17):
    """Constructs a Botsort model
    Args:
        fps (float): Frames per second of the video
        with_reid (bool): Whether to use reid
        cmc_method (str): CMC method to use, either orb or sift
        cmc_downscale (int): Downscale factor for CMC. Higher values are faster but less accurate
        mot_year (int): Pretrained model for MOT17 or MOT20
    """
    if mot_year not in (17, 20):
        raise ValueError(f'MOT year must be 17 or 20, got {mot_year}')
    if cmc_method not in ('orb', 'sift'):
        raise ValueError(f'CMC method must be orb or sift, got {cmc_method}')
    
    conf = _default_config()
    conf.with_reid = with_reid
    conf.cmc_method = cmc_method
    conf.cmc_downscale = cmc_downscale

    if conf.with_reid:
        url = f'https://github.com/ashwhall/BoT-SORT/releases/download/{VERSION}/mot{mot_year}_sbs_S50.pth'
        torch.hub.load_state_dict_from_url(url, progress=True, map_location=torch.device('cpu'))

        conf.fast_reid_config = f'fast_reid/configs/MOT{mot_year}/sbs_S50.yml'
        conf.fast_reid_weights = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(url))
    
    model = BoTSORT(conf, frame_rate=fps)
    wrapper = Wrapper(model)
    return wrapper

