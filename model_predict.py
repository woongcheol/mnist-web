from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np

# Model configuration
img_width, img_height = 28, 28

# 데이터 준비
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Reshaping    
input_train = input_train.reshape((input_train.shape[0], img_width, img_height, 1))
input_test = input_test.reshape((input_test.shape[0], img_width, img_height, 1))
input_shape = (img_width, img_height, 1)

# 정규화
input_train = input_train.astype('float32') / 255.0
input_test = input_test.astype('float32') / 255.0

# 모델 로드
loaded_model = load_model('h5_model.h5')

sample_index = 0
sample_input, sample_target = input_test[sample_index], target_test[sample_index]
sample_input_array = np.array([sample_input])
predictions = loaded_model.predict(sample_input_array)
prediction = np.argmax(predictions[0])
print(f'실제값: {sample_target} - 예측값: {prediction}')

# 이미지 입력
import cv2

sample_image = input_train[0]
sample_image.reshape(28, 28)

# 결과
img = cv2.imwrite( "mnist_sample.jpg", (sample_image + 0.5) * 255)