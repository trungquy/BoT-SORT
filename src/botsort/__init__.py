from dataclasses import dataclass
from pathlib import Path
import torch
from botsort.tracker.bot_sort import BoTSORT

DIR = Path(__file__).parent

WEIGHTS_VERSION = 'v0.1.0'


@dataclass
class BotSortConfig:
    fps: float
    with_reid: bool = False
    cmc_method: str = 'orb'
    cmc_downscale: int = 2
    cmc_im_threshold: bool = False
    mot_year: str = '17'
    device: str = 'cpu'
    fp16: bool = True
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 70
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 1.6
    min_box_area: float = 10
    fuse_score: bool = True
    proximity_thresh: float = 0.5
    fuse: bool = True
    trt: bool = False
    appearance_thresh: float = 0.25
    fast_reid_config: str = None
    fast_reid_weights: str = None

    @property
    def mot20(self):
        return not self.fuse_score
    @property
    def name(self):
        return None
    @property
    def ablation(self):
        return None

    def validate(self):
        if self.fps <= 0:
            raise ValueError(f'fps must be greater than 0, got {self.fps}')
        if self.mot_year not in ('17', '20'):
            raise ValueError(f'MOT year must be \'17\' or \'20\', got {self.mot_year}')
        if self.cmc_method not in ('orb', 'sift', 'ecc'):
            raise ValueError(f'CMC method must be one of {{orb, sift, ecc}}, got {self.cmc_method}')
        if self.cmc_downscale < 1 or not type(self.cmc_downscale) == int:
            raise ValueError(f'CMC downscale must be a positive integer, got {self.cmc_downscale}')


def construct_botsort(conf: BotSortConfig):
    """Constructs a Botsort model
    Args:
        fps (float): Frames per second of the video
        with_reid (bool): Whether to use reid
        cmc_method (str): CMC method to use, either orb or sift
        cmc_downscale (int): Downscale factor for CMC. Higher values are faster but less accurate
        mot_year (str): Pretrained model for MOT17 or MOT20, either '17' or '20'
        device (str): Device to use, 'cpu', 'cuda', etc.
    """
    conf.validate()
    if conf.with_reid:
        url = f'https://github.com/ashwhall/BoT-SORT/releases/download/{WEIGHTS_VERSION}/mot{conf.mot_year}_sbs_S50.pth'
        torch.hub.load_state_dict_from_url(url, progress=True, map_location=torch.device('cpu'))

        conf.fast_reid_config = str(DIR / 'fast_reid' / 'configs' / f'MOT{conf.mot_year}' / 'sbs_S50.yml')
        conf.fast_reid_weights = str(Path(torch.hub.get_dir()) / 'checkpoints' / Path(url).name)
    
    return BoTSORT(conf, frame_rate=conf.fps)
