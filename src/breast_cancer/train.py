from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam


def train_model(
    model: tf.keras.Model,
    ds_train: tf.data.Dataset,
    ds_val: tf.data.Dataset,
    experiment_name: str,
    training_cfg: dict,
    dir_model_logs: Any,
    epochs: int = 5,
    use_csv_logger: bool = True,
    use_early_stopping: bool = False,
) -> tf.keras.callbacks.History:
    """Compile and train a Keras model with callbacks and logging.

    Args:
        model: Keras model to train.
        ds_train: Training dataset.
        ds_val: Validation dataset.
        experiment_name: Name of the experiment for log files.
        training_cfg: Dict containing training hyperparameters (e.g., lr, callbacks configs).
        dir_model_logs: Path object for saving logs and checkpoints.
        epochs: Number of epochs to train (default 5).
        use_csv_logger: Whether to log training history to CSV.
        use_early_stopping: Whether to apply early stopping.
    Returns:
        History: Training history object from model.fit().
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    # Clear any previous session
    print("[1/5] Clearing previous session.")
    tf.keras.backend.clear_session()

    # Define optimizer, loss, and metrics
    print("[2/5] Defining optimizer, loss, and metrics.")
    optimizer, loss, metrics = get_compile_components(training_cfg["lr"])

    # Load and compile model
    print("[3/5] Compiling model.")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Set callbacks
    print("[4/5] Setting up callbacks.")
    (dir_model_logs / model.name).mkdir(exist_ok=True)
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", **training_cfg["reduce_on_plateau"]),
    ]
    if use_csv_logger:
        callbacks.append(
            CSVLogger(dir_model_logs / model.name / f"initial_{experiment_name}.csv")
        )
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val_loss", **training_cfg["early_stopping"])
        )

    # Train the model and return history
    print("[5/5] Starting training process.")
    history = model.fit(
        ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks
    )

    print("Training complete...")
    return history


def get_compile_components(
    lr: float,
) -> Tuple[tf.keras.optimizers.Optimizer, tf.keras.losses.Loss, list]:
    """Return optimizer, loss, and metrics for model compilation.

    Args:
        lr: Learning rate for optimizer.
    Returns:
        Tuple containing optimizer, loss, and list of metrics.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    optimizer = Adam(learning_rate=lr)
    loss = BinaryCrossentropy()
    metrics = [AUC(), BinaryAccuracy(), Precision(), Recall()]

    return optimizer, loss, metrics
