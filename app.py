from flask import Flask, render_template, request
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import base64
from io import BytesIO

app = Flask(__name__)

# Cargar el modelo y el procesador de imágenes (redimensionamiento, normalización, etc.)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
# Cargar el modelo preentrenado
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_data = None
    image_format = None

    if request.method == "POST":
        if "image" in request.files:
            image_file = request.files["image"] 
            image = Image.open(image_file).convert("RGB")

            # Procesar la imagen para clasificación
            inputs = processor(images=image, return_tensors="pt") #divide la imagen en parches y la prepara para el modelo
            outputs = model(**inputs) #Se pasa por el vision transformer

            logits = outputs.logits 
            probabilities = torch.nn.functional.softmax(logits, dim=-1) # Convertir logits a probabilidades 

            predicted_class_idx = probabilities.argmax(-1).item() # Obtener el índice de la clase con mayor probabilidad

            result = model.config.id2label[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()

            # Convertir imagen a base64 para mostrarla en el HTML
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data = img_str
            image_format = "jpeg"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_data=image_data,
        image_format=image_format
    )

if __name__ == "__main__":
    app.run(debug=True)