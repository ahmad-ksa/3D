"""Run by tests/test_video.py in a fresh process: the real LatentSync pipeline + worker.cmd_lipsync with tiny
random models (same structure), a MediaPipe stand-in, ffmpeg by plain name, a folder with spaces + Arabic."""
import json, os, subprocess, sys, wave
from pathlib import Path
import numpy as np, torch
from studio3d.video import worker, media
config = worker._setup_paths()
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.whisper.model import ModelDimensions, Whisper
from latentsync.whisper.audio2feature import Audio2Feature
import latentsync.utils.image_processor as ip

root = Path(sys.argv[1])
torch.manual_seed(0)
# tiny models with LatentSync's structure (13 input channels, audio cross-attention, temporal layers)
dims = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=32, n_audio_head=2, n_audio_layer=1, n_vocab=64,
                       n_text_ctx=8, n_text_state=32, n_text_head=2, n_text_layer=1)
wpath = root / "tiny.pt"
torch.save({"dims": dims.__dict__, "model_state_dict": Whisper(dims).state_dict()}, wpath)
full = OmegaConf.load(config.PACKAGE_DIR / "vendor_video" / "configs" / "unet" / "stage2.yaml")
m = OmegaConf.to_container(full.model)
m.update(block_out_channels=[32, 64], down_block_types=["CrossAttnDownBlock3D", "DownBlock3D"],
         up_block_types=["UpBlock3D", "CrossAttnUpBlock3D"], layers_per_block=1, cross_attention_dim=32,
         norm_num_groups=8, motion_module_resolutions=[1, 2], sample_size=8)
m["motion_module_kwargs"]["num_attention_heads"] = 2
ckpt = root / "unet.pt"
torch.save({"state_dict": UNet3DConditionModel.from_config(m).state_dict()}, ckpt)
cfg = OmegaConf.create({"data": {"num_frames": 16, "resolution": 64, "audio_feat_length": [2, 2]}})

def tiny_lipsync(_config):
    sch = DDIMScheduler.from_pretrained(str(config.PACKAGE_DIR / "vendor_video" / "configs"))
    enc = Audio2Feature(model_path=str(wpath), device="cpu", num_frames=16, audio_feat_length=[2, 2])
    vae = AutoencoderKL(block_out_channels=[8, 16], down_block_types=["DownEncoderBlock2D"] * 2,
                        up_block_types=["UpDecoderBlock2D"] * 2, latent_channels=4, norm_num_groups=8)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(m, str(ckpt), device="cpu", dtype=torch.float32)
    return LipsyncPipeline(vae=vae, audio_encoder=enc, unet=unet, scheduler=sch).to("cpu"), cfg, torch.float32

worker._load_lipsync = tiny_lipsync

class FakeFaces:      # MediaPipe stand-in: face in the middle, lost in every 3rd frame
    def __init__(self, device="cpu"):
        self.n = 0
    def __call__(self, frame):
        self.n += 1
        if self.n % 3 == 0:
            return None, None
        h, w = frame.shape[:2]
        lmk = np.zeros((106, 2))
        lmk[[43, 48, 49, 50, 51]] = (w * 0.40, h * 0.35)
        lmk[101:106] = (w * 0.60, h * 0.35)
        lmk[[74, 77, 83, 86]] = (w * 0.50, h * 0.55)
        return (0, 0, w, h), lmk

ip.FaceDetector = FakeFaces
scene = root / "مشهد 1 with space"
scene.mkdir()
subprocess.run([media.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                "testsrc=s=160x96:r=24:d=1.5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(scene / "motion.mp4")], check=True)
t = np.arange(int(2.6 * 16000)) / 16000            # voice LONGER than the clip -> clip is looped
with wave.open(str(scene / "voice.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes((0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16).tobytes())
os.environ["PATH"] = str(media.ffmpeg_dir(root / "bin")) + os.pathsep + os.environ["PATH"]
job = {"index": 1, "video": str(scene / "motion.mp4"), "audio": str(scene / "voice.wav"),
       "out": str(scene / "talk.mp4"), "steps": 2}
cwd = os.getcwd()
r = worker.cmd_lipsync({"jobs": [job]}, config)
assert os.getcwd() == cwd
assert r["items"][0]["ok"], r
out = scene / "talk.mp4"
info = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-i", str(out)], capture_output=True, text=True).stderr
assert "Audio:" in info and "Video:" in info, info          # the voice is muxed in
assert abs(media.duration(out) - 2.6) < 0.25, media.duration(out)

# no face in any frame: this scene is reported, the command does not crash
ip.FaceDetector = type("NoFace", (), {"__init__": lambda s, device="cpu": None,
                                      "__call__": lambda s, f: (None, None)})
job2 = dict(job, out=str(scene / "talk2.mp4"))
r2 = worker.cmd_lipsync({"jobs": [job2, job]}, config)
assert r2["items"][0].get("no_face") and r2["items"][1]["ok"] is False
print("LIPSYNC-OK")
