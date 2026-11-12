"""Pose skeletons (OpenPose style) used to steer text-to-image toward a rig-friendly pose.

The skeleton is only a guide image for the image model (ControlNet); it is never drawn on the
character. Front view, full body, centred, feet near the bottom.
"""
import math

from PIL import Image, ImageDraw

POSES = ("free", "t", "a")

# OpenPose body-18 order
# 0 nose 1 neck 2 r_sho 3 r_elb 4 r_wri 5 l_sho 6 l_elb 7 l_wri 8 r_hip 9 r_knee 10 r_ank
# 11 l_hip 12 l_knee 13 l_ank 14 r_eye 15 l_eye 16 r_ear 17 l_ear
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
         (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
          (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
          (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def keypoints(pose: str):
    """Normalised (x, y) in 0..1, image coords (y down). The character's right is image-left."""
    # body proportions of a standing adult, about 7.5 heads tall, filling ~84% of the height
    top, bottom = 0.08, 0.92
    hgt = bottom - top
    head = hgt / 7.5
    cx = 0.5
    neck_y = top + 1.15 * head
    sho_y = neck_y + 0.25 * head
    sho = 0.95 * head            # half shoulder width
    hip_y = top + 3.9 * head
    hip = 0.6 * head
    knee_y = top + 5.75 * head
    ank_y = bottom - 0.15 * head
    arm = 3.2 * head             # shoulder -> wrist
    if pose == "t":
        ang = 0.0                # straight out
    else:                        # A-pose: about 45 degrees down
        ang = math.radians(45)
    dx, dy = math.cos(ang), math.sin(ang)
    r_elb = (cx - sho - 0.5 * arm * dx, sho_y + 0.5 * arm * dy)
    r_wri = (cx - sho - arm * dx, sho_y + arm * dy)
    l_elb = (cx + sho + 0.5 * arm * dx, sho_y + 0.5 * arm * dy)
    l_wri = (cx + sho + arm * dx, sho_y + arm * dy)
    nose_y = top + 0.6 * head
    eye_y = top + 0.45 * head
    return [
        (cx, nose_y), (cx, neck_y),
        (cx - sho, sho_y), r_elb, r_wri,
        (cx + sho, sho_y), l_elb, l_wri,
        (cx - hip, hip_y), (cx - hip * 0.95, knee_y), (cx - hip * 0.9, ank_y),
        (cx + hip, hip_y), (cx + hip * 0.95, knee_y), (cx + hip * 0.9, ank_y),
        (cx - 0.18 * head, eye_y), (cx + 0.18 * head, eye_y),
        (cx - 0.4 * head, eye_y + 0.1 * head), (cx + 0.4 * head, eye_y + 0.1 * head),
    ]


def skeleton_image(pose: str, size: int = 1024) -> Image.Image:
    """OpenPose-style skeleton on black (what the pose ControlNet was trained on)."""
    if pose not in ("t", "a"):
        raise ValueError(pose)
    pts = [(x * size, y * size) for x, y in keypoints(pose)]
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    width = max(4, size // 100)
    for i, (a, b) in enumerate(LIMBS):
        c = tuple(int(v * 0.6) for v in COLORS[i % len(COLORS)])
        d.line([pts[a], pts[b]], fill=c, width=width)
    r = max(4, size // 120)
    for i, (x, y) in enumerate(pts):
        rr = r * 1.6 if i in (0, 14, 15, 16, 17) else r   # nose / eyes / ears: front view
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=COLORS[i])
    return img


PROMPT_HINT = {
    # full body, FRONT view facing the camera (face visible), T-pose, arms straight out
    "t": "，正面视角，面向镜头，看着镜头，脸部清晰，全身，T字姿势，双臂水平伸直，站立",
    "a": "，正面视角，面向镜头，看着镜头，脸部清晰，全身，A字姿势，双臂自然斜向下张开，站立",
}

# the image model tends to draw T-posed people from behind: forbid it explicitly
NEGATIVE_HINT = ",背面,背影,背对镜头,后视图,看不到脸"
