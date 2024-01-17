import tensorflow_hub as hub

# URL del modelo de MobileNet para detección de objetos
model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

# Cargar el modelo
model = hub.load(model_url)
