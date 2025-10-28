import time
from src.loader import SystemLoader, TrainingProgressBar, JupyterLoader, TFTrainingLoader

# ====================================================
# 1️⃣ TEST: SYSTEM LOADER (CLI / Terminal)
# ====================================================
def test_system_loader():
    loader = SystemLoader("Daten werden geladen", total_steps=50)
    for i in range(50):
        time.sleep(0.05)  # Simuliert Arbeit
        loader.update(i + 1, f"📦 Datei {i+1}/50")
    loader.done("Alle Dateien geladen")


# ====================================================
# 2️⃣ TEST: TRAINING PROGRESS BAR (für Offline-Training)
# ====================================================
def test_training_progressbar():
    bar = TrainingProgressBar(total_epochs=20)
    for epoch in range(20):
        time.sleep(0.1)
        logs = {"accuracy": 0.5 + 0.02 * epoch, "loss": 0.7 - 0.01 * epoch}
        bar.update(epoch + 1, logs)


# ====================================================
# 3️⃣ TEST: JUPYTER LOADER (Nur in Notebook sichtbar)
# ====================================================
def test_jupyter_loader():
    try:
        loader = JupyterLoader("Notebook Loading", total=30)
        for i in range(30):
            time.sleep(0.05)
            loader.update(i + 1, f"Schritt {i+1}")
        loader.done()
    except Exception as e:
        print("⚠️ JupyterLoader kann nur in Notebooks korrekt angezeigt werden:", e)


# ====================================================
# 4️⃣ TEST: TF TRAINING LOADER (Simulation)
# ====================================================
def test_tf_training_loader():
    import tensorflow as tf
    import numpy as np

    # Dummy-Daten für Mini-Training
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=(100,))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Verwende unseren Loader als Callback
    loader_callback = TFTrainingLoader(total_epochs=5)
    model.fit(X, y, epochs=5, batch_size=16, callbacks=[loader_callback], verbose=0)


# ====================================================
# MAIN TEST EXECUTION
# ====================================================
if __name__ == "__main__":
    print("\n==============================")
    print("🧩 TEST: SYSTEM LOADER")
    print("==============================")
    test_system_loader()

    print("\n==============================")
    print("🧩 TEST: TRAINING PROGRESS BAR")
    print("==============================")
    test_training_progressbar()

    print("\n==============================")
    print("🧩 TEST: TF TRAINING LOADER")
    print("==============================")
    test_tf_training_loader()

    print("\n✅ Alle Loader erfolgreich getestet.\n")
