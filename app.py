from flask import Flask, request, jsonify
from flask_cors import CORS

from food_recognition import *

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/api/process-image", methods=["POST"])
def process_image():
    print('processing image!!!')
    print("files: ", request.files)
    if "photo" not in request.files:
        print("no photo present")
        return jsonify({"error": "No file part"})

    photo = request.files["photo"]
    
    img_path = "temp_image.png"
    photo.save(img_path)

    print("tot hier gaat het goed")
    food_item = main(img_path)
    print("food item: ", food_item)
    return jsonify({'food_item': food_item})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

# app.run()
