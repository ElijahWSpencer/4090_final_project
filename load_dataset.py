import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_size = (224, 224)
batch_size = 32

# Used to train dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\Elijah\\Desktop\\animals\\train",
    image_size=img_size,
    batch_size=batch_size
)

# Used to validate dataset
val_ds = keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\Elijah\\Desktop\\animals\\val",
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

base_model = keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=img_size + (3,))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("ews_animal_distinguisher_v3.keras")

