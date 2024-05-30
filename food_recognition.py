import tensorflow
from keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

from train_model import prepare_data_generators
from nutrional_value import nutritional_value, nutrition_weights, food_weights

trained = True

if trained:
    print("Loading model")
    model = load_model("food101_model.keras")
else:
    model = MobileNetV2(weights="imagenet")


# Functie om een afbeelding voor te bereiden
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_food(img_path):
    img_array = prepare_image(img_path)
    preds = model.predict(img_array)
    # Decodeer de voorspellingen (top 3)
    results = decode_predictions(preds, top=3)[0]
    print(results)
    return results


def predict_food_101(img_path):
    img_array = prepare_image(img_path)
    preds = model.predict(img_array)
    train_generator, test_generator = prepare_data_generators()
    class_indices = train_generator.class_indices
    class_indices = {v: k for k, v in class_indices.items()}  # Omgekeerde mapping
    predicted_class = np.argmax(preds, axis=1)[0]
    food_item = class_indices[predicted_class]
    return food_item


nutrition_db = {
    "apple": {"calories": 52, "carbs": 14, "protein": 0.3, "fat": 0.2},
    "banana": {"calories": 96, "carbs": 27, "protein": 1.3, "fat": 0.3},
}


def get_nutrition_info(food_item):
    return nutrition_db.get(food_item.lower(), "Nutrition info not available")


def find_nutritional_content(food_name):
    if food_name.lower() in nutritional_value:
        food_info = nutritional_value[food_name.lower()]
        nutritional_content = {}
        
        for nutrient, values in food_info["per_100g"].items():
            per_100g_unit = nutrition_weights.get(nutrient, "")
            nutritional_content[nutrient] = {"per_100g": (values, per_100g_unit)}

        for nutrient, values in food_info["per_item"].items():
                per_item_unit = nutrition_weights.get(nutrient, "")
                nutritional_content[nutrient]["per_item"] = (values, per_item_unit)

        return nutritional_content
    else:
        return None


def display_nutritional_value(food_item, nutritional_info):
    if nutritional_info:
        weight = food_weights[food_item.lower()]
        print(f"Nutritional content for {food_item.capitalize()}:")
        for nutrient, values in nutritional_info.items():
            per_100g_value, per_100g_unit = values["per_100g"]
            per_item_value, per_item_unit = values["per_item"]
            print(f"{nutrient.capitalize()} per 100g: {per_100g_value} {per_100g_unit}")
            print(f"{nutrient.capitalize()} per item: {round(per_item_value / weight * 100, 2)} {per_item_unit}")

    else:
        print("Nutritional information not found for the specified food.")


def main(img_path):

    if trained:
        food_item = predict_food_101(img_path)
        print(f"Food item: {food_item}")
        nutrition = find_nutritional_content(food_item)
        display_nutritional_value(food_item, nutrition)
    else:
        predictions = predict_food(img_path)
        for pred in predictions:
            food_item, confidence = pred[1], pred[2]
            print(f"Predicted: {food_item} with confidence: {confidence:.2f}")
            nutrition_info = get_nutrition_info(food_item)
            print(f"Nutrition info: {nutrition_info}")

img_path = "Images/image3.jpg"
main(img_path)

