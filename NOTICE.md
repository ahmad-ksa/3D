# Third-party notices

## Tencent Hunyuan3D 2.0
Studio3D downloads and runs the Tencent Hunyuan3D 2.0 models and uses the `hy3dgen`
library from https://github.com/Tencent-Hunyuan/Hunyuan3D-2 (pinned commit
`f8db63096c8282cb27354314d896feba5ba6ff8a`).

Licensed under the **TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT**
(https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE). Key points:

- The license **does not apply in the European Union, the United Kingdom and South Korea**;
  use there is not licensed.
- If the licensee's products/services had more than **1 million monthly active users**
  on the release date, a separate license from Tencent is required.
- Use must follow Tencent's Acceptable Use Policy.

`studio3d/vendor/custom_rasterizer` is an independent pure-PyTorch re-implementation of the
API of Hunyuan3D's `custom_rasterizer` CUDA extension, written to avoid shipping
compiled, unsigned binaries.

Tencent Hunyuan is a trademark of Tencent; Studio3D is not affiliated with Tencent.

## HunyuanDiT (text-to-image, optional)
`Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled` — Tencent Hunyuan Community License.

## Other libraries
PyTorch (BSD), Gradio (Apache-2.0), trimesh (MIT), PyMeshLab (GPL-3.0, used as a separate
library), rembg (MIT), diffusers / transformers (Apache-2.0).
