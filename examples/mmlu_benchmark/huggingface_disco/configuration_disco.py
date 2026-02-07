# Copyright 2025 MASEval contributors. DISCO predictor config for Hugging Face Hub.

# Use configuration_utils so this works when transformers uses lazy top-level imports
from transformers.configuration_utils import PretrainedConfig


class DiscoConfig(PretrainedConfig):
    """Configuration for DISCO predictor (PCA + Random Forest) on the Hub."""

    model_type = "disco"

    def __init__(
        self,
        n_components: int = 256,
        sampling_name: str = "",
        number_item: str = "",
        fitted_model_type: str = "",
        use_full_prompt: bool = True,
        data_path: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.sampling_name = sampling_name
        self.number_item = number_item
        self.fitted_model_type = fitted_model_type
        self.use_full_prompt = use_full_prompt
        self.data_path = data_path
