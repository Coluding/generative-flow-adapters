from __future__ import annotations

from generative_flow_adapters.models.base.interfaces import ModuleBackboneWrapper, infer_prediction_type


class DiffusersUNetWrapper(ModuleBackboneWrapper):
    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        pretrained_model_name_or_path: str,
        subfolder: str | None = None,
        prediction_type: str | None = None,
    ) -> "DiffusersUNetWrapper":
        try:
            from diffusers import UNet2DConditionModel, UNet2DModel
        except ImportError as exc:
            raise RuntimeError("Install the optional diffusers dependencies to use pretrained wrappers.") from exc

        prediction = infer_prediction_type(model_type, prediction_type)
        loaders = [UNet2DConditionModel, UNet2DModel]
        last_error: Exception | None = None
        for loader in loaders:
            try:
                module = loader.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
                return cls(module=module, model_type=model_type, prediction_type=prediction)
            except Exception as exc:  # pragma: no cover - soft dependency path
                last_error = exc
        raise RuntimeError(f"Failed to load diffusers UNet from {pretrained_model_name_or_path}") from last_error
