Third-party code vendored into this repository lives here.

Contents:
- `lvdm/`: copied from the DynamiCrafter code vendored inside the AVID repository
- `avid_utils/`: copied from the AVID repository utility package

Credit:
- AVID: "Adapting Video Diffusion Models to World Models"
- DynamiCrafter: original latent video diffusion backbone used by AVID

These files are kept under `external_deps` to make the ownership boundary explicit. Local framework code should treat them as vendored dependencies rather than first-party adapter code.
