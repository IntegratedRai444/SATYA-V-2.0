from flask_cors import CORS
from flask import Flask, jsonify

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({"apiUrl": "http://localhost:5002"})

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image_endpoint():
    # ... existing code ...
    return jsonify({'error': 'No image data provided'}), 400

@app.route('/api/ai/analyze/image', methods=['POST'])
def analyze_image_alias():
    return analyze_image_endpoint()

if __name__ == "__main__":
    app.run(port=5002) 