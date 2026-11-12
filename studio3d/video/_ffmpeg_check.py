"""Imported by the Smart App Control check: fails if Windows blocks the bundled ffmpeg.exe."""
import subprocess

import imageio_ffmpeg

subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], check=True, capture_output=True, timeout=120)
