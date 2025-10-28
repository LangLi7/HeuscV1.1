"""
Hybrid CNN-LSTM model builder for HEUSC AI Trading.
Builds a convolutional + recurrent model fully configurable via JSON.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers, optimizers

DEFAULT_SETTINGS_PATH = Path("config/settings.json")


class SettingsNotFoundError(FileNotFoundError):
    """Raised when the configuration file cannot be located."""


@dataclass
class ModelConfig:
    """Container for the model specific portion of the configuration file."""

    raw: Dict[str, Any]

    @property
    def cnn(self) -> Dict[str, Any]:
        return self.raw.get("cnn", {})

    @property
    def lstm(self) -> Dict[str, Any]:
        return self.raw.get("lstm", {})

    @property
    def dense(self) -> Dict[str, Any]:
        return self.raw.get("dense", {})

    @property
    def optimizer(self) -> Dict[str, Any]:
        return self.raw.get("optimizer", {})

    @property
    def loss(self) -> str:
        return self.raw.get("loss", "binary_crossentropy")

    @property
    def metrics(self) -> Sequence[str]:
        return self.raw.get("metrics", ["accuracy"])


def load_settings(settings_path: Path | str = DEFAULT_SETTINGS_PATH) -> Dict[str, Any]:
    """Load a JSON settings file."""
    settings_path = Path(settings_path)
    if not settings_path.exists():
        raise SettingsNotFoundError(f"Config file not found: {settings_path.resolve()}")
    with settings_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _configure_optimizer(optimizer_config: Dict[str, Any]) -> optimizers.Optimizer:
    """Instantiate an optimizer based on the configuration block."""
    optimizer_type = optimizer_config.get("type", "adam").lower()
    learning_rate = optimizer_config.get("learning_rate", 1e-3)
    decay = optimizer_config.get("decay", 0.0)

    if optimizer_type == "adam":
        return optimizers.Adam(learning_rate=learning_rate, decay=decay)
    if optimizer_type == "rmsprop":
        return optimizers.RMSprop(learning_rate=learning_rate, decay=decay)
    if optimizer_type == "sgd":
        momentum = optimizer_config.get("momentum", 0.0)
        nesterov = optimizer_config.get("nesterov", False)
        return optimizers.SGD(
            learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov
        )
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def _ensure_length(values: Sequence[Any], expected: int, name: str) -> List[Any]:
    """Validate list lengths, repeating the last element if required."""
    if not values:
        raise ValueError(f"Configuration '{name}' must not be empty.")
    if len(values) == expected:
        return list(values)
    if len(values) > expected:
        return list(values[:expected])
    padded = list(values)
    while len(padded) < expected:
        padded.append(padded[-1])
    return padded


def build_hybrid_cnn_lstm(
    input_shape: Sequence[int],
    settings: Optional[Dict[str, Any]] = None,
    settings_path: Path | str = DEFAULT_SETTINGS_PATH,
) -> Model:
    """Create and compile the CNN-LSTM model described in settings.json."""
    if settings is None:
        settings = load_settings(settings_path)

    model_config = ModelConfig(settings.get("model", {}))
    cnn_cfg = model_config.cnn
    lstm_cfg = model_config.lstm
    dense_cfg = model_config.dense

    cnn_layers = int(cnn_cfg.get("layers", 0))
    lstm_layers = int(lstm_cfg.get("layers", 0))
    dense_layers = int(dense_cfg.get("layers", 0))

    model_layers: List[layers.Layer] = []

    # --- CNN feature extractor ---
    if cnn_layers <= 0:
        raise ValueError("At least one CNN layer must be defined.")

    filters = _ensure_length(cnn_cfg.get("filters", [64]), cnn_layers, "cnn.filters")
    kernel_sizes = _ensure_length(cnn_cfg.get("kernel_sizes", [3]), cnn_layers, "cnn.kernel_sizes")
    activations = _ensure_length(cnn_cfg.get("activations", ["relu"]), cnn_layers, "cnn.activations")
    pooling_type = cnn_cfg.get("pooling", "max").lower()
    pool_size = int(cnn_cfg.get("pool_size", 2))
    dropout_rate = float(cnn_cfg.get("dropout", 0.0))

    for idx in range(cnn_layers):
        layer_kwargs = {
            "filters": int(filters[idx]),
            "kernel_size": int(kernel_sizes[idx]),
            "activation": activations[idx],
            "padding": "same",
        }
        if idx == 0:
            layer_kwargs["input_shape"] = tuple(input_shape)
        model_layers.append(layers.Conv1D(**layer_kwargs))

        if pooling_type == "max":
            model_layers.append(layers.MaxPooling1D(pool_size=pool_size))
        elif pooling_type == "avg":
            model_layers.append(layers.AveragePooling1D(pool_size=pool_size))
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        if dropout_rate > 0:
            model_layers.append(layers.Dropout(dropout_rate))

    # --- LSTM sequence modeling ---
    if lstm_layers <= 0:
        raise ValueError("At least one LSTM layer must be defined.")

    lstm_units = _ensure_length(lstm_cfg.get("units", [64]), lstm_layers, "lstm.units")
    return_sequences = _ensure_length(
        lstm_cfg.get("return_sequences", [False]), lstm_layers, "lstm.return_sequences"
    )
    recurrent_dropout = float(lstm_cfg.get("recurrent_dropout", 0.0))
    lstm_dropout = _ensure_length(lstm_cfg.get("dropout", [0.0]), lstm_layers, "lstm.dropout")
    bidirectional = bool(lstm_cfg.get("bidirectional", False))

    for idx in range(lstm_layers):
        lstm = layers.LSTM(
            int(lstm_units[idx]),
            return_sequences=bool(return_sequences[idx]),
            dropout=float(lstm_dropout[idx]),
            recurrent_dropout=recurrent_dropout,
        )
        if bidirectional:
            model_layers.append(layers.Bidirectional(lstm))
        else:
            model_layers.append(lstm)

    # --- Dense output head ---
    dense_units = _ensure_length(dense_cfg.get("units", [1]), dense_layers, "dense.units")
    dense_activations = _ensure_length(
        dense_cfg.get("activations", ["sigmoid"]), dense_layers, "dense.activations"
    )

    for idx in range(dense_layers):
        model_layers.append(layers.Dense(int(dense_units[idx]), activation=dense_activations[idx]))

    keras_model = tf.keras.Sequential(model_layers, name="hybrid_cnn_lstm")
    optimizer = _configure_optimizer(model_config.optimizer)
    keras_model.compile(optimizer=optimizer, loss=model_config.loss, metrics=list(model_config.metrics))
    return keras_model


def build_callbacks(training_config: Dict[str, Any], logs_root: Path) -> List[callbacks.Callback]:
    """Create a list of standard callbacks based on the training configuration."""
    callbacks_list: List[callbacks.Callback] = []
    early_cfg = training_config.get("early_stopping", {})

    if early_cfg.get("enabled", False):
        callbacks_list.append(
            callbacks.EarlyStopping(
                monitor=early_cfg.get("monitor", "val_loss"),
                patience=int(early_cfg.get("patience", 10)),
                restore_best_weights=True,
            )
        )

    if training_config.get("tensorboard", False):
        tensorboard_logdir = logs_root / "tensorboard"
        tensorboard_logdir.mkdir(parents=True, exist_ok=True)
        callbacks_list.append(callbacks.TensorBoard(log_dir=str(tensorboard_logdir)))

    return callbacks_list


__all__ = [
    "SettingsNotFoundError",
    "ModelConfig",
    "build_hybrid_cnn_lstm",
    "build_callbacks",
    "load_settings",
]
