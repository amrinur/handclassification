from google.colab import files
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Mengunggah gambar ke Colab
uploaded = files.upload()

# Mendapatkan nama file yang diunggah
file_name = list(uploaded.keys())[0]

# Membuat path file
img_path = file_name

# Memuat gambar dan melakukan pre-processing
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Memprediksi kelas gambar menggunakan model
prediction = model.predict(img_array)

# Mendapatkan label kelas dengan nilai tertinggi
predicted_class = np.argmax(prediction)

# Menampilkan hasil prediksi
class_labels = ['Gunting', 'Batu', 'kertas']

plt.imshow(img)
plt.axis('off')
plt.title(f'Prediksi: {class_labels[predicted_class]}')
plt.show()
