import tensorflow as tf
from keras import models
from keras.layers import Rescaling, MaxPooling2D,  Conv2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# 1. 設定參數與路徑
data_dir = "weather_dataset" 
batch_size = 16
img_size = (320, 320)
seed = 123

# 2. 載入資料集（80% 訓練 / 20% 驗證）
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

# 3. 標準化圖片像素（0~1）
normalization_layer = Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 4. 建立 CNN 模型
model = models.Sequential([
  
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(320, 320, 3)),
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

# 5. 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. 訓練模型
history = model.fit(train_ds, validation_data = val_ds, epochs = 10)

# 7. 顯示訓練結果
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# 8. 顯示驗證集中的 6 張圖片與預測結果
for images, labels in val_ds.take(1):  
    predictions = model.predict(images)
    predicted_classes = tf.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(6):  # 顯示前 6 張圖
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy())
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted_classes[i]]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")
    plt.show()
    break

# 9. 印出最後一輪的準確率與損失
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"訓練準確率：{final_train_acc:.4f}")
print(f"驗證準確率：{final_val_acc:.4f}")

