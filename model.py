import tensorflow as tf
from keras import models
from keras.layers import MaxPooling2D, Rescaling, Conv2D, Flatten, Dense
import matplotlib.pyplot as plt

# 1. 設定參數與路徑
data_dir = "weather_dataset"  # ← 請替換成你的資料資料夾
batch_size = 16
img_size = (256, 256)
seed = 111

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
    Conv2D(64, 3, activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(256, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# 5. 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. 訓練模型
history = model.fit(train_ds, batch_size = 64, validation_data = val_ds, epochs = 10)

# 7. 顯示訓練結果
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# 8. 顯示驗證集中的 4 張圖片與預測結果
for images, labels in val_ds.take(1):  # 取一個 batch
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

