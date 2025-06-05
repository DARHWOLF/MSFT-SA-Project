import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
# 2) FootprintTracker (time→kWh→kgCO₂e)
# ───────────────────────────────────────────────────────────────────────────────
class FootprintTracker:
    def __init__(self, cpu_w=50, gpu_w=150, ef=0.5):
        self.power_kw = (cpu_w + gpu_w)/1000.0
        self.ef = ef
        self._t0 = None
    def start(self):   self._t0 = time.time()
    def stop(self):
        hrs = (time.time() - self._t0)/3600.0
        return hrs * self.power_kw * self.ef

# ───────────────────────────────────────────────────────────────────────────────
# 3) Tiny MNIST subset (500 train / 100 val / 100 test)
# ───────────────────────────────────────────────────────────────────────────────
(xf,yf),(xt,yt) = tf.keras.datasets.mnist.load_data()
x = xf[:600].astype('float32')/255.0; y = yf[:600]
x = x[...,None]
x_train, x_val = x[:500], x[500:]; y_train, y_val = y[:500], y[500:]
x_test = (xt[:100].astype('float32')/255.0)[...,None]; y_test = yt[:100]

# ───────────────────────────────────────────────────────────────────────────────
# 4) Model builder
# ───────────────────────────────────────────────────────────────────────────────
def build_cnn():
    m = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax'),
    ])
    return m

# ───────────────────────────────────────────────────────────────────────────────
# 5) Train baseline
# ───────────────────────────────────────────────────────────────────────────────
baseline = build_cnn()
baseline.compile('adam','sparse_categorical_crossentropy',['accuracy'])

tracker = FootprintTracker(); tracker.start()
hist_base = baseline.fit(x_train,y_train,
                         validation_data=(x_val,y_val),
                         epochs=5, verbose=2)
em_base = tracker.stop()
_, acc_base = baseline.evaluate(x_test, y_test, verbose=0)
params = baseline.count_params()

# ───────────────────────────────────────────────────────────────────────────────
# 6) Manual magnitude pruning (50%) + fine‐tune with mixed precision
# ───────────────────────────────────────────────────────────────────────────────
# 6.1 Extract & prune weights
pruned_weights = []
for w in baseline.get_weights():
    if w.ndim>1:  # only prune weight matrices, not biases
        thresh = np.percentile(np.abs(w), 50)
        w = w * (np.abs(w) >= thresh)
    pruned_weights.append(w)

# 6.2 Build a fresh model, load pruned weights
pruned_model = build_cnn()
pruned_model.set_weights(pruned_weights)

# 6.3 Compile with mixed precision & do one fine‐tune epoch
tf.keras.mixed_precision.set_global_policy('mixed_float16')
pruned_model.compile('adam','sparse_categorical_crossentropy',['accuracy'])

tracker = FootprintTracker(); tracker.start()
# one epoch to recover
hist_pruned = pruned_model.fit(x_train,y_train,
                               validation_data=(x_val,y_val),
                               epochs=2, verbose=2)
em_pruned = tracker.stop()
_, acc_pruned = pruned_model.evaluate(x_test, y_test, verbose=0)
params_pruned = pruned_model.count_params()

# ───────────────────────────────────────────────────────────────────────────────
# 7) Plotting (7 polished figures in one window)
# ───────────────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(3, 3, figsize=(18, 12))  # 3x3 grid (7 used, 2 empty)

# 1) Loss curves
ax = axs[0, 0]
ax.plot(hist_base.history['loss'],     '-o', label='Baseline Train')
ax.plot(hist_base.history['val_loss'], '-o', label='Baseline Val')
ax.plot(hist_pruned.history['loss'],     's', label='Pruned+MP Train')
ax.plot(hist_pruned.history['val_loss'], 's', label='Pruned+MP Val')
style_ax(ax, "Loss Curves", "Epoch", "Loss")
ax.legend(fontsize=12)

# 2) Accuracy curves
ax = axs[0, 1]
ax.plot(hist_base.history['accuracy'],     '-o', label='Baseline Train')
ax.plot(hist_base.history['val_accuracy'], '-o', label='Baseline Val')
ax.plot(hist_pruned.history['accuracy'],     's', label='Pruned+MP Train')
ax.plot(hist_pruned.history['val_accuracy'], 's', label='Pruned+MP Val')
style_ax(ax, "Accuracy Curves", "Epoch", "Accuracy")
ax.legend(fontsize=12)

# 3) Emissions bar
ax = axs[0, 2]
ax.bar(['Baseline','Pruned+MP'], [em_base, em_pruned])
style_ax(ax, "Estimated Emissions (kg CO₂e)", "Model", "kg CO₂e")

# 4) Test Accuracy bar
ax = axs[1, 0]
ax.bar(['Baseline','Pruned+MP'], [acc_base, acc_pruned])
style_ax(ax, "Test Accuracy", "Model", "Accuracy")

# 5) Parameter count bar
ax = axs[1, 1]
ax.bar(['Baseline','Pruned+MP'], [params, params_pruned])
style_ax(ax, "Model Size (# Parameters)", "Model", "# Params")

# 6) Emissions vs Accuracy scatter
ax = axs[1, 2]
ax.scatter(em_base, acc_base, s=120, label='Baseline')
ax.scatter(em_pruned, acc_pruned, s=120, label='Pruned+MP')
style_ax(ax, "Emissions vs Accuracy", "kg CO₂e", "Accuracy")
ax.legend(fontsize=12)

# 7) Histogram of Pruned Conv Weights
ax = axs[2, 0]
w0 = pruned_weights[0].flatten()
ax.hist(w0, bins=30)
style_ax(ax, "Pruned Conv Layer Weights", "Weight Value", "Frequency")

# Turn off the two unused subplots
axs[2, 1].axis('off')
axs[2, 2].axis('off')

plt.tight_layout()
plt.show()

