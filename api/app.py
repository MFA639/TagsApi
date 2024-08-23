from flask import Flask
import torch
import numpy as np

app = Flask(__name__)

model_path = "models/model.pth"
model = torch.load(model_path)
model.eval()

mlb_classes_path = "models/mlb_classes.npy"
mlb_classes = np.load(mlb_classes_path, allow_pickle=True)

@app.route('/')
def index():
    return "Model loaded successfully!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

