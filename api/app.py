from flask import Flask, request, jsonify
import torch
import numpy as np
import tensorflow_hub as hub

app = Flask(__name__)

model_path = "models/model.pth"
model = torch.load(model_path)
model.eval()

mlb_classes_path = "models/mlb_classes.npy"
mlb_classes = np.load(mlb_classes_path, allow_pickle=True)

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def predict_tags(model, text, use_model, mlb_classes, top_n=5):
    title_embedding = use_model([text]).numpy()
    title_tensor = torch.tensor(title_embedding, dtype=torch.float32)

    with torch.no_grad():
        y_pred_tensor = model(title_tensor)
        y_pred_scores = y_pred_tensor.numpy().flatten()

    predicted_tags_with_scores = {mlb_classes[i]: float(y_pred_scores[i]) for i in range(len(mlb_classes))}
    predicted_tags_with_scores = dict(sorted(predicted_tags_with_scores.items(), key=lambda item: item[1], reverse=True))
    
    top_predicted_tags = {k: round(predicted_tags_with_scores[k], 4) for k in list(predicted_tags_with_scores)[:top_n]}
    return top_predicted_tags

@app.route('/')
def index():
    return "Model loaded successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    predictions = predict_tags(model, text, use_model, mlb_classes, top_n=5)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

