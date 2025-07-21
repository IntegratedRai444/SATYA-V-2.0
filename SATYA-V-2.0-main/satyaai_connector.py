import os
import hashlib
import requests
import datetime
from pathlib import Path
from server.python import deepfake_analyzer

class SatyaAIConnector:
    def __init__(self, api_url=None):
        self.api_url = api_url or "http://localhost:5002/api/analyze"
        self.cache = {}

    def file_hash(self, file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def analyze_file(self, file_path, file_type, mode='auto'):
        file_hash = self.file_hash(file_path)
        if file_hash in self.cache:
            return self.cache[file_hash]

        result = None
        if mode in ['api', 'auto']:
            try:
                result = self.analyze_via_api(file_path, file_type)
            except Exception as e:
                print(f"API analysis failed: {e}")
                if mode == 'api':
                    raise
        if result is None and mode in ['local', 'auto']:
            result = self.analyze_locally(file_path, file_type)
        if result:
            self.cache[file_hash] = result
        return result

    def analyze_via_api(self, file_path, file_type):
        endpoint = f"{self.api_url}/{file_type}/ai"
        with open(file_path, 'rb') as f:
            files = {file_type: f}
            response = requests.post(endpoint, files=files)
        response.raise_for_status()
        data = response.json()
        data['filename'] = os.path.basename(file_path)
        data['file_type'] = file_type
        data['analysis_date'] = datetime.datetime.now().isoformat()
        return data

    def analyze_locally(self, file_path, file_type):
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        # Use file hash as seed for deterministic results
        file_hash = self.file_hash(file_path)
        seed = int(file_hash[:8], 16)
        import random
        random.seed(seed)
        if file_type == 'image':
            return deepfake_analyzer.analyze_image(file_bytes)
        elif file_type == 'video':
            return deepfake_analyzer.analyze_video(file_bytes, filename=os.path.basename(file_path))
        elif file_type == 'audio':
            return deepfake_analyzer.analyze_audio(file_bytes, filename=os.path.basename(file_path))
        else:
            raise ValueError("Unsupported file type")

# Example usage:
if __name__ == "__main__":
    connector = SatyaAIConnector(api_url="http://localhost:5002/api/analyze")
    result = connector.analyze_file("test.jpg", "image", mode="auto")
    print(result) 