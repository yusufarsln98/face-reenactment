# Third-party source copies

Raw upstream source, kept for comparison with the ports in this project.
No project code reads these files.

## basicsr_v1.4.2/ (arch_util.py, img_util.py)
- Upstream: XPixelGroup/BasicSR, version v1.4.2.  Licence: Apache-2.0.
- Source URL: none recorded in the two files; the upstream tag is
  https://github.com/XPixelGroup/BasicSR/tree/v1.4.2.
- Backs: the ported `default_init_weights` in scripts/video/gfpgan_arch.py,
  which the GFPGAN restoration step in the report depends on.

## gfpgan_probes/ (13 files: v0.2.4, master and v1.3.8 variants)
- Upstream: TencentARC/GFPGAN, versions v0.2.4 and v1.3.8, plus the master
  snapshot.  Licence: Apache-2.0.
- Source URL: https://github.com/TencentARC/GFPGAN, recorded in
  README_v1.3.8.md; inference_gfpgan.py and its v1.3.8 copy record the
  Real-ESRGAN model URL.  The other files record no source URL.
- Backs: the report finding that the GFPGAN fidelity weight is inert ("the
  official inference path passes no weight argument"): the v1.3.8 utils add
  the `weight` keyword that v1.3.4_utils.py, kept in .agent/docs for the
  cited evidence path, does not pass to the model.
