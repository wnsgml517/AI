#ref: https://www.tensorflow.org/tutorials/images/classification
import tensorflow as tf
from tensorflow.keras.layers   import Input, Conv2D, MaxPool2D, Dense  
from tensorflow.keras.layers   import BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# 모델 만들기
model = VGG16(weights=None, include_top=True, classes=3, input_shape=(244,244,3))
model.summary()

##model.summary()

#3: 이미지 정규화
#3-1:
train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    rotation_range=20,    
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2)

test_datagen = ImageDataGenerator(rescale= 1./255)

#데이터 불러오기
img_width, img_height = 224, 224
train_dir= "C:/Users/wnsgm/Desktop/checktree/train/train"
test_dir = "C:/Users/wnsgm/Desktop/checktree/test/test"
train_generator= train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height), batch_size=32,
    class_mode="categorical", subset='training')
valid_generator= train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height), batch_size=32,
    class_mode="categorical", subset='validation')

test_generator= test_datagen.flow_from_directory(
    test_dir, target_size=(img_width, img_height), batch_size=32,
    class_mode="categorical")

print("train_generator.class_indices=", train_generator.class_indices)
print("test_generator.class_indices=", test_generator.class_indices)

print("train_generator.classes.shape=", train_generator.classes.shape)
print("valid_generator.classes.shape=", valid_generator.classes.shape)
print("test_generator.classes.shape=",  test_generator.classes.shape)

train_steps= int(np.ceil(train_generator.classes.shape[0]/train_generator.batch_size))
valid_steps= int(np.ceil(valid_generator.classes.shape[0]/valid_generator.batch_size))
test_steps= int(np.ceil(test_generator.classes.shape[0]/test_generator.batch_size))
print("train_steps=",train_steps)
print("valid_steps=",valid_steps)
print("test_steps=",test_steps)

#4: generator을 이용하여 모델 학습
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ret = model.fit(train_generator, epochs=10,  
                validation_data=valid_generator,
                steps_per_epoch= train_steps,
                validation_steps=valid_steps,
                verbose=2)

#5:  
#5-1: 컨퓨전 매트릭스
y_pred = model.predict(train_generator, steps=train_steps, verbose=2)
y_label = np.argmax(y_pred, axis = 1)
C = tf.math.confusion_matrix(train_generator.labels, y_label)
print("confusion_matrix(C):", C)

#5: 평가
train_loss, train_acc = model.evaluate(train_generator,
                                       steps = train_steps,
                                       verbose=2)
test_loss, test_acc = model.evaluate(test_generator,
                                     steps = test_steps,
                                     verbose=2)

#6: 출력
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'],  "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['accuracy'],     "b-", label="train accuracy")
ax[1].plot(ret.history['val_accuracy'], "r-", label="val_accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
plt.legend(loc="best")
fig.tight_layout()
plt.show()

