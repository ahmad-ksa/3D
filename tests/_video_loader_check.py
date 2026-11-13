"""Run by tests/test_video.py in a fresh process (video diffusers first on the path, like the real worker):
a tiny FastWan-shaped model folder on disk goes through the worker's real loading and generation code."""
import gc, json, sys, torch, numpy as np
from pathlib import Path
from studio3d.video import worker
worker._setup_paths()
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from transformers import UMT5Config, UMT5EncoderModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from PIL import Image

root = Path(sys.argv[1])
snap = root / "snap"
torch.manual_seed(0)
# --- a tiny FastWan-like snapshot (same layout as the real repo) ---
(snap / "scheduler").mkdir(parents=True)
(snap / "model_index.json").write_text(json.dumps({
    "_class_name": "WanDMDPipeline", "_diffusers_version": "0.35.0", "expand_timesteps": True,
    "scheduler": ["diffusers", "UniPCMultistepScheduler"], "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"], "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"]}))
(snap / "scheduler" / "scheduler_config.json").write_text(json.dumps(
    {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 8.0}))
tr = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8, in_channels=4,
                           out_channels=4, text_dim=16, freq_dim=16, ffn_dim=32, num_layers=2, rope_max_seq_len=32)
tr.to(torch.bfloat16).save_pretrained(snap / "transformer", max_shard_size="40KB")   # sharded like the real one
vae = AutoencoderKLWan(base_dim=8, z_dim=4, in_channels=12, out_channels=12, patch_size=2, dim_mult=[1, 1, 1, 1],
                       num_res_blocks=1, temperal_downsample=[False, True, True], latents_mean=[0.0] * 4,
                       latents_std=[1.0] * 4, is_residual=True, scale_factor_spatial=16)
vae.save_pretrained(snap / "vae")
te_cfg = UMT5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=2, num_heads=2,
                    relative_attention_num_buckets=8, feed_forward_proj="gated-gelu")
te_ref = UMT5EncoderModel(te_cfg).eval()
te_ref.save_pretrained(snap / "text_encoder", max_shard_size="20KB")
vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
for w in "a cat waves the girl smiles and sings slowly".split():
    vocab.setdefault(w, len(vocab))
tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>", eos_token="</s>").save_pretrained(
    snap / "tokenizer")

# 1. the 8-bit loader gives exactly what from_pretrained + layerwise casting gives, in 8-bit storage
ref = WanTransformer3DModel.from_pretrained(snap / "transformer", torch_dtype=torch.bfloat16).eval()
ref.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
pipe = worker._load_i2v(snap)
mine = pipe.transformer
assert pipe.config.expand_timesteps and pipe.scheduler.config.prediction_type == "flow_prediction"
assert pipe.scheduler.config.use_flow_sigmas and pipe.scheduler.config.flow_shift == 8.0
assert mine.blocks[0].ffn.net[0].proj.weight.dtype == torch.float8_e4m3fn
assert mine.blocks[0].norm2.weight.dtype == torch.float32                  # kept in fp32 like from_pretrained
assert mine.condition_embedder.time_embedder.linear_1.weight.dtype == torch.float32
for (n1, p1), (n2, p2) in zip(sorted(ref.state_dict().items()), sorted(mine.state_dict().items())):
    assert n1 == n2 and p1.dtype == p2.dtype and torch.equal(p1.float(), p2.float()), n1
x = torch.randn(1, 4, 2, 8, 8, dtype=torch.bfloat16)
t = torch.full((1,), 500.0)
e = torch.randn(1, 7, 16, dtype=torch.bfloat16)
with torch.no_grad():
    y_ref = ref(x, t, e, return_dict=False)[0]
    y = mine(x, t, e, return_dict=False)[0]
    assert torch.equal(y_ref, y)
    # 8-bit storage + block streaming (CPU stand-in for the GPU): same result, weights back in 8-bit
    from diffusers.hooks import apply_group_offloading
    apply_group_offloading(mine, onload_device=torch.device("cpu"), offload_type="block_level",
                           num_blocks_per_group=1, low_cpu_mem_usage=True)
    assert torch.equal(mine(x, t, e, return_dict=False)[0], y)
    assert mine.blocks[1].ffn.net[0].proj.weight.dtype == torch.float8_e4m3fn

# 2. text encoder: 8-bit storage, close to the bf16 reference, and really freed after use
tok, te = worker._load_text_encoder(snap)
assert te.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype == torch.float8_e4m3fn
assert te.shared.weight.dtype == torch.bfloat16
ids = torch.tensor([[3, 4, 5, 1, 0, 0]]); m = (ids > 0).long(); m[0, 3] = 1
with torch.no_grad():
    a = te(ids, m).last_hidden_state.float()
    b = te_ref.to(torch.bfloat16)(ids, m).last_hidden_state.float()
assert torch.isfinite(a).all()
cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
assert cos > 0.97, cos
del tok, te, a, b

class Cfg:
    MODELS_DIR = root / "models"
    VIDEO_MODEL = "x/y"
embs = worker._prompt_embeds(Cfg, snap, ["a cat waves", "the girl smiles", "a cat waves"])
alive = [o for o in gc.get_objects() if isinstance(o, UMT5EncoderModel) and o is not te_ref]
assert not alive, "text encoder still in memory after encoding"
assert len(list((root / "models" / "video" / "prompt_cache").glob("*.pt"))) == 2
assert embs["a cat waves"].shape == (1, 512, 16)

# 3. the whole i2v command: two scenes in one process, one model load
img = root / "in put صورة.png"
Image.new("RGBA", (80, 48), (120, 30, 200, 0)).save(img)
worker._download = lambda *a, **k: snap
out1, out2 = root / "مشهد 1" / "motion.mp4", root / "مشهد 2" / "motion.mp4"
out1.parent.mkdir(); out2.parent.mkdir()
jobs = [{"index": 1, "image": str(img), "prompt": "a cat waves", "width": 64, "height": 64, "frames": 9,
         "steps": 2, "fps": 24, "out": str(out1)},
        {"index": 2, "image": str(img), "prompt": "she sings slowly", "width": 64, "height": 64, "frames": 5,
         "steps": 2, "fps": 24, "out": str(out2)}]
loads = []
orig = worker._load_i2v
worker._load_i2v = lambda *a, **k: loads.append(1) or orig(*a, **k)
r = worker.cmd_i2v({"jobs": jobs}, Cfg)
assert len(loads) == 1 and [i["frames"] for i in r["items"]] == [9, 5]
assert out1.stat().st_size > 0 and out2.stat().st_size > 0
print("LOADER-OK")
