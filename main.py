# 1. 掛載 Google Drive
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# 2. 讀取你的共享資料夾
import glob
import os

# 設定你的資料夾路徑
folder_path = '/content/drive/Shareddrives/lipo_stamps_data/'

# 讀取所有 jpg, jpeg, png 檔案
image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + \
              glob.glob(os.path.join(folder_path, '*.jpeg')) + \
              glob.glob(os.path.join(folder_path, '*.png'))

print(f"✅ 找到 {len(image_paths)} 張圖片")

# 3. 載入必要套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# 4. 載入 MobileNetV2 模型（不包含最後分類層）
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# 5. 圖片前處理 function
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 6. 把 Drive 上的圖片都先提取特徵，建成特徵庫
features = []

for _ in range(100):
   for img_path in image_paths:
      img_array = preprocess_img(img_path)
      feature_vector = model.predict(img_array)
      features.append(feature_vector[0])  # 拿掉 batch 維度

features = np.array(features)
print(len(features[7]))

print("✅ 特徵提取完成！")


# ==============================
# 🎯 這裡開始是搜尋功能！
# ==============================

# 7. 上傳要搜尋的 Query 圖片
from google.colab import files

uploaded = files.upload()

query_path = list(uploaded.keys())[0]  # 只拿第一張圖
print(f"🔍 搜尋圖片：{query_path}")

# 8. 處理上傳的 Query 圖片
query_array = preprocess_img(query_path)
query_vector = model.predict(query_array)

# 9. 計算 Query 與資料庫每張圖的相似度
similarities = cosine_similarity(query_vector, features)[0]  # 結果是 1D array

# 10. 找最像的前5張圖
top_k = 5
top_k_indices = similarities.argsort()[::-1][:top_k]

print(f"🔎 找到最相似的前 {top_k} 張圖片：")
for idx in top_k_indices:
    print(f"{os.path.basename(image_paths[idx % len(image_paths)])} (相似度: {similarities[idx]:.4f})")

# 11. 顯示搜尋結果
plt.figure(figsize=(15, 5))

# 顯示 Query 圖片
plt.subplot(1, top_k + 1, 1)
query_img = image.load_img(query_path)
plt.imshow(query_img)
plt.title("Query Image")
plt.axis('off')

# 顯示最像的圖片
for i, idx in enumerate(top_k_indices):
    plt.subplot(1, top_k + 1, i + 2)
    matched_img = image.load_img(image_paths[idx % len(image_paths)])
    plt.imshow(matched_img)
    plt.title(f"Top {i+1}\n{similarities[idx]:.2f}")
    plt.axis('off')

plt.show()
