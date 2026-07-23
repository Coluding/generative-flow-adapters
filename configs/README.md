# configs/ — experiment configs, grouped by backbone provider

| Dir | Provider | Contents |
|---|---|---|
| `wan22/` | `wan2.2` (Wan2.2-TI2V-5B) | All Wan2.2 experiment configs: AVID AdaLN/xattn adapters (gatelow / replace / overfit triangle), DC-UNet capacity arms, the ACWM push_block config, and the action-free `flow_wan22_shortcut_only_metaworld.yaml` |
| `wan21/` | `wan2.1` | Wan2.1 shortcut / AVID-shortcut runs and the generic Wan output adapter |
| `dynamicrafter/` | `dynamicrafter` | DynamiCrafter-base configs: AVID shortcut arms (direct / affine / action), HyperAlign, UniCon, multimodal |
| `opensora/` | `opensora` | OpenSora output adapter |
| `dev/` | `dummy` | Test/smoke configs with no real backbone (used by unit tests) |
| `base/` | — | Backbone/adapter tier definitions referenced via `*_config_path` (unchanged location) |
| `prompts/` | — | Text-prompt tables + precomputed context embeddings (unchanged location) |

Conventions:

- Filenames keep the flat-era naming (`{model_type}_{backbone}_{adapter}_{variant}_{dataset}.yaml`), so
  vault notes citing a config by filename stay greppable.
- `base/` and `prompts/` paths are referenced from inside configs
  (`wan_config_path`, `unet_config_path`, `text_prompts_file`) and did not move.
- Cluster job scripts in `jobs/experiments_cluster/` reference these paths;
  the script → config → ticket map lives in `jobs/experiments_cluster/README.md`.
