import re
from pathlib import Path
import torch
from botsort.tracker.bot_sort import BoTSORT

DIR = Path(__file__).parent

# Load the version from setup.py
version = None
with open(DIR.parent.parent / 'setup.py', 'r') as f:
    for line in f:
        if 'version=' in line:
            # Greedily match between quotes with \=["'](.*)["']
            version = re.search(r'=["\'](.*)["\']', line).group(1)
            VERSION = f'v{version}'
            break


def _default_config():
    class DotDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
        def copy(self):
            return DotDict(super(DotDict, self).copy())

    return DotDict(
        device = 'cpu',
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


def construct_botsort(fps: float, with_reid: bool, cmc_method: str = 'orb', cmc_downscale: int = 4, mot_year: int = 17, device='cpu'):
    """Constructs a Botsort model
    Args:
        fps (float): Frames per second of the video
        with_reid (bool): Whether to use reid
        cmc_method (str): CMC method to use, either orb or sift
        cmc_downscale (int): Downscale factor for CMC. Higher values are faster but less accurate
        mot_year (int): Pretrained model for MOT17 or MOT20
        device (str): Device to use, 'cpu', 'cuda', etc.
    """
    if mot_year not in (17, 20):
        raise ValueError(f'MOT year must be 17 or 20, got {mot_year}')
    if cmc_method not in ('orb', 'sift', 'ecc'):
        raise ValueError(f'CMC method must be one of {{orb, sift, ecc}}, got {cmc_method}')
    
    conf = _default_config()
    conf.device = device
    conf.with_reid = with_reid
    conf.cmc_method = cmc_method
    conf.cmc_downscale = cmc_downscale

    if conf.with_reid:
        url = f'https://github.com/ashwhall/BoT-SORT/releases/download/{VERSION}/mot{mot_year}_sbs_S50.pth'
        torch.hub.load_state_dict_from_url(url, progress=True, map_location=torch.device('cpu'))

        conf.fast_reid_config = str(DIR / 'fast_reid' / 'configs' / f'MOT{mot_year}' / 'sbs_S50.yml')
        conf.fast_reid_weights = str(Path(torch.hub.get_dir()) / 'checkpoints' / Path(url).name)
    
    return BoTSORT(conf, frame_rate=fps)


def _test_botsort():
    import numpy as np

    botsort = construct_botsort(30, True, device='cuda')

    detections = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    image = np.zeros((768, 768, 3), dtype=np.uint8)

    print('Out:', botsort.update(detections, image))
