import tensorflow as tf
from keras import models
from keras.layers import Rescaling, MaxPooling2D, Conv2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 參數設定
data_dir = "weather_dataset"
batch_size = 64
img_size = (120, 120)
seed = 123

# 2. 載入資料集（訓練/驗證 8:2）
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("類別名稱：", class_names)

# 3. 僅標準化（不使用資料增強）
normalization_layer = Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 4. 效能優化
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 5. 建立 CNN 模型
model = models.Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(120, 120, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# 6. 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 訓練模型
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# 8. 顯示訓練過程圖
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# 9. 顯示驗證集預測結果
for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    predicted_classes = tf.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy())
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted_classes[i]]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")
    plt.show()
    break

# 10. 訓練集混淆矩陣
y_true_train = []
y_pred_train = []

for images, labels in train_ds.unbatch().batch(batch_size):
    preds = model.predict(images)
    y_true_train.extend(labels.numpy())
    y_pred_train.extend(tf.argmax(preds, axis=1).numpy())

cm_train = confusion_matrix(y_true_train, y_pred_train)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
disp_train.plot(cmap='Blues')
plt.title("Training Set Confusion Matrix")
plt.show()

# 11. 驗證集混淆矩陣
y_true_val = []
y_pred_val = []

for images, labels in val_ds.unbatch().batch(batch_size):
    preds = model.predict(images)
    y_true_val.extend(labels.numpy())
    y_pred_val.extend(tf.argmax(preds, axis=1).numpy())

cm_val = confusion_matrix(y_true_val, y_pred_val)
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=class_names)
disp_val.plot(cmap='Blues')
plt.title("Validation Set Confusion Matrix")
plt.show()

# 12. 顯示最後準確率
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"訓練準確率：{final_train_acc:.4f}")
print(f"驗證準確率：{final_val_acc:.4f}")
