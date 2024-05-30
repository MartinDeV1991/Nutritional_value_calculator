from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint


def prepare_data_generators():
    train_data_dir = "Training_set"
    test_data_dir = "Test_set"

    IMAGE_SIZE = (100, 100)
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    print("class indices: ", train_generator.class_indices)
    return train_generator, test_generator


def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(100, 100, 3)
    )
    base_model.trainable = False
    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model():
    train_generator, test_generator = prepare_data_generators()

    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)

    checkpoint = ModelCheckpoint(
        "fruit_model.keras", monitor="val_accuracy", save_best_only=True, mode="max"
    )

    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=10,
        callbacks=[checkpoint],
    )
    return model, train_generator

# model, train_generator = train_model()

