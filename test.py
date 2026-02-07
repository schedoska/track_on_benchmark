import torch
from model.trackon_predictor import Predictor
import imageio.v3 as iio
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

from utils.train_utils import load_args_from_yaml
model_args = load_args_from_yaml('config/test_dinov2.yaml')

model = Predictor(model_args, checkpoint_path="trackon2_dinov2_checkpoint.pt", support_grid_size=0).to(device).eval()
model = torch.compile(model)
model.reset()  # reset internal memory before a new video

# video:   (1, T, 3, H, W)
# queries: (1, N, 3) with rows = (t, x, y)


queries = torch.tensor([[
    [1., 304., 190.],
]],device=device)
frames = iio.imread('vid2.mp4', plugin="FFMPEG")  # plugin="pyav"
frames_small = [cv2.resize(f, (400, 400)) for f in frames]
print('tensoring...')
video = torch.from_numpy(np.stack(frames_small)) \
              .permute(0, 3, 1, 2) \
              .unsqueeze(0).float()
#first_frame = video[0, 0].permute(1, 2, 0)  # C,H,W -> H,W,C

model.eval()

for t in range(video.shape[1]):
    if t % 3 == 0:
        continue
    #frame = video[:, t]  # (1, 3, H, W)
    vid_batch = video[0,t].unsqueeze(0).to(device)
    # Add queries whose start time is t
    new_queries = (
        queries[0, queries[0, :, 0] == t, 1:]
        if queries is not None else None
    )
    tstart = time.time()
    print('beg')
    # Track a single frame
    with torch.no_grad():
        points_t, vis_t = model.forward_frame(
            vid_batch,
            new_queries=new_queries
        )

    print(time.time() - tstart)
    print(points_t)

