import tensorflow as tf
from keras import models
from keras.layers import MaxPooling2D, Rescaling, Conv2D, Flatten, Dense
import matplotlib.pyplot as plt

# 1. 設定參數與路徑
data_dir = "weather_dataset"  # 替換為你的資料夾路徑
batch_size = 16
img_size = (256, 256)
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
    # 第一層：卷積層 (Convolutional Layer)
    Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),  # 32 個 3x3 卷積核
    MaxPooling2D(),  # 池化層 (MaxPooling)

    # 第二層：卷積層 (Convolutional Layer)
    Conv2D(64, 3, activation='relu'),  # 64 個 3x3 卷積核
    MaxPooling2D(),  # 池化層

    # 第三層：卷積層 (Convolutional Layer)
    Conv2D(128, 3, activation='relu'),  # 128 個 3x3 卷積核
    MaxPooling2D(),  # 池化層

    # 展平層 (Flatten)
    Flatten(),  # 展開為一維陣列

    # 全連接層 (Fully Connected Layer)
    Dense(128, activation='relu'),  # 128 個神經元
    Dense(len(class_names), activation='softmax')  # 輸出層，使用 softmax 激活函數
])

# 5. 編譯模型
model.compile(
    optimizer='adam',  # 優化器
    loss='sparse_categorical_crossentropy',  # 損失函數（適用於多分類）
    metrics=['accuracy']  # 評估指標：準確率
)

# 6. 訓練模型
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

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
    for i in range(4):  # 取前 4 張圖片
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(images[i].numpy())  # 顯示圖片
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted_classes[i]]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")
    plt.show()  # 確保圖片正確顯示
    break
