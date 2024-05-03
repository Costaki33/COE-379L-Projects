# Family Facial Recognition for Home Security Applications

The goal of this project is to create a machine learning model that can help us identify a familiar face at a residence or a stranger who may be a source of suspicion. The model will be trained on pictures of Costaki’s family members who live at his house as well as a dataset of strangers who do not live at his house. Our goal is for our models to accurately and efficiently tell us who the specific family member is or if it is a stranger.

## Models

We will be creating two machine learning models—a convolutional neural network (CNN) and a convolutional neural network with LeNet-5 architecture—to test our data on them. We chose a standard convolutional neural network to show the simplest form of neural network architecture and its limitations, and chose a more refined LeNet-5 model to improve our result accuracy as it is an extremely strong model in image recognition.

### Standard Convolutional Neural Network

```Model: "sequential_12"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_30 (Conv2D)              │ (None, 248, 248, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_27 (MaxPooling2D) │ (None, 124, 124, 32)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_31 (Conv2D)              │ (None, 122, 122, 64)   │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_28 (MaxPooling2D) │ (None, 61, 61, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_32 (Conv2D)              │ (None, 59, 59, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_29 (MaxPooling2D) │ (None, 29, 29, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_10 (Flatten)            │ (None, 107648)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_21 (Dense)                │ (None, 512)            │    55,116,288 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_9 (Dropout)             │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_22 (Dense)                │ (None, 5)              │         2,565 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 55,212,103 (210.62 MB)
 Trainable params: 55,212,101 (210.62 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B) (reg CNN)
```

### LeNet-5 Convolutional Neural Network Architecture
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_33 (Conv2D)              │ (None, 248, 248, 6)    │           168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d_2             │ (None, 124, 124, 6)    │             0 │
│ (AveragePooling2D)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_34 (Conv2D)              │ (None, 122, 122, 16)   │           880 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d_3             │ (None, 61, 61, 16)     │             0 │
│ (AveragePooling2D)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_11 (Flatten)            │ (None, 59536)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_23 (Dense)                │ (None, 120)            │     7,144,440 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_24 (Dense)                │ (None, 84)             │        10,164 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_25 (Dense)                │ (None, 5)              │           425 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 7,156,077 (27.30 MB)
 Trainable params: 7,156,077 (27.30 MB)
 Non-trainable params: 0 (0.00 B)
```

### Testing

After the two models run in the Jupyter Notebook (`Family_Facial_Recognition.ipynb`), there is a cell at the bottom of each model in the code as follows:
```
# Evaluate the model
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {val_acc}")

# Save the model
model.save('family_member_classifier.h5')

# Load and use the model
from tensorflow.keras.models import load_model
model = load_model('family_member_classifier.h5')

# Predicting with the model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load an image
img = image.load_img('test_images/<IMAGE NAME>', target_size=(250, 250))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Make a prediction
prediction = model.predict(img_tensor)
print(prediction)
```
To test an image from the `test_images` folder, replace `<IMAGE NAME>` with the name of an image. The output is in the form

`[[athena, costaki, george, stranger, teresa]]`

where each name gets a value from 0 to 1 based on who the model thinks the image is of.
