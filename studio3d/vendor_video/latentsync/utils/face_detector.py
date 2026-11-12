"""Studio3D: insightface needs a locally compiled extension (blocked by Smart App Control) and its
face models are for non-commercial research only. This drop-in replacement uses Google MediaPipe
(Apache-2.0) and returns the same (bbox, 106-point landmarks) that LatentSync expects."""
from studio3d.video.face_mp import FaceDetector  # noqa: F401
