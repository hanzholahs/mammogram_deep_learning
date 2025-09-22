import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam

from breast_cancer.models.binary_average_vote import BinaryAverageVote
from breast_cancer.models.binary_majority_vote import BinaryMajorityVote


def train_model(
    model,
    ds_train,
    ds_val,
    experiment_name,
    training_cfg,
    dir_model_logs,
    epochs=5,
    use_csv_logger = True,
    use_early_stopping=False,
):
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
        callbacks.append(CSVLogger(dir_model_logs / model.name / f"initial_{experiment_name}.csv"))
    if use_early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", **training_cfg["early_stopping"]))

    # Train the model and return history
    print("[5/5] Starting training process.")
    history = model.fit(
        ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks
    )

    print("Training complete...")
    return history

def get_compile_components(lr):
    optimizer = Adam(learning_rate=lr)
    loss = BinaryCrossentropy()
    metrics = [AUC(), BinaryAccuracy(), Precision(), Recall()]

    return optimizer, loss, metrics