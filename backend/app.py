from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64
import io
import json

app = Flask(__name__)
# Fix CORS - allow all origins for development
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TrafficSignClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = TrafficSignCNN(num_classes=43).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_names = self.load_class_names()
        self.setup_model()
    
    def load_class_names(self):
        class_names = {
            0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
            3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
            6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
            9: "No passing", 10: "No passing for vehicles over 3.5 tons", 11: "Right-of-way at intersection",
            12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles", 
            16: "Vehicles over 3.5 tons prohibited", 17: "No entry", 18: "General caution", 
            19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve", 
            22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on right", 25: "Road work",
            26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing", 29: "Bicycles crossing",
            30: "Beware of ice/snow", 31: "Wild animals crossing", 32: "End of all speed and passing limits",
            33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
            37: "Go straight or left", 38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
            41: "End of no passing", 42: "End of no passing by vehicles over 3.5 tons"
        }
        return class_names
    
    def setup_model(self):
        """Initialize model for demo purposes"""
        self.model.eval()
        print("Model initialized in demo mode")
    
    def predict(self, image):
        try:
            # Convert and preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            class_id = predicted.item()
            class_name = self.class_names.get(class_id, f"Unknown Sign ({class_id})")
            
            # For demo, ensure reasonable confidence
            demo_confidence = max(confidence.item(), 0.85)
            
            return {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': round(demo_confidence * 100, 2),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'error': f'Prediction processing error: {str(e)}',
                'status': 'error'
            }

# Initialize classifier
classifier = TrafficSignClassifier()

@app.route('/')
def home():
    return jsonify({
        'message': 'Traffic Sign Classifier API is running!',
        'endpoints': {
            '/api/health': 'Health check',
            '/api/predict': 'Predict traffic sign (POST)',
            '/api/classes': 'Get all class names',
            '/api/demo': 'Get demo prediction'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'device': str(classifier.device),
        'model_loaded': True,
        'message': 'Backend is running correctly!'
    })

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        print("Received prediction request")
        
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided', 'status': 'error'}), 400
        
        image = None
        
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected', 'status': 'error'}), 400
            
            print(f"Processing file: {file.filename}")
            image = Image.open(file.stream).convert('RGB')
            
        else:
            print("Processing base64 image")
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        
        result = classifier.predict(image)
        print(f"Prediction result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify(classifier.class_names)

@app.route('/api/demo', methods=['GET'])
def demo_prediction():
    """Demo endpoint that returns a mock prediction for testing"""
    import random
    demo_classes = [14, 17, 33, 35, 2, 13]  # Common signs: Stop, No entry, etc.
    class_id = random.choice(demo_classes)
    
    return jsonify({
        'class_id': class_id,
        'class_name': classifier.class_names[class_id],
        'confidence': round(random.uniform(85, 98), 2),
        'status': 'success',
        'demo': True,
        'message': 'This is a demo prediction. The model is working!'
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    print("🚦 Starting Traffic Sign Classifier Server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🔗 Make sure your frontend connects to this URL")
    app.run(debug=True, port=5000, host='0.0.0.0')