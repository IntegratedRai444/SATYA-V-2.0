"""
Test Data Factory

Generates consistent test data for all test types.
"""
from faker import Faker
from faker.providers import internet, file, lorem, person, python
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import tempfile
import shutil
from typing import Dict, Any, List, Union, Optional

class TestDataFactory:
    """Factory for generating test data"""
    
    def __init__(self):
        self.fake = Faker()
        self.fake.add_provider(internet)
        self.fake.add_provider(file)
        self.fake.add_provider(lorem)
        self.fake.add_provider(person)
        self.fake.add_provider(python)
        
        # Create a temporary directory for test files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="satya_test_"))
        
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_user(self, **overrides) -> Dict[str, Any]:
        """Generate a test user"""
        user = {
            "id": self.fake.uuid4(),
            "username": self.fake.user_name(),
            "email": self.fake.email(),
            "first_name": self.fake.first_name(),
            "last_name": self.fake.last_name(),
            "is_active": True,
            "is_verified": True,
            "created_at": self.fake.date_time_this_year().isoformat(),
            "updated_at": self.fake.date_time_this_year().isoformat(),
        }
        user.update(overrides)
        return user
    
    def create_test_analysis_result(self, **overrides) -> Dict[str, Any]:
        """Generate a test analysis result"""
        result = {
            "id": self.fake.uuid4(),
            "user_id": self.fake.uuid4(),
            "file_name": f"test_{self.fake.file_name(extension='jpg')}",
            "file_type": "image/jpeg",
            "file_size": random.randint(1000, 1000000),
            "is_fake": random.choice([True, False]),
            "confidence": round(random.uniform(0.5, 1.0), 2),
            "analysis_time": round(random.uniform(0.1, 5.0), 2),
            "model_version": "1.0.0",
            "metadata": {
                "width": 800,
                "height": 600,
                "detection_regions": [
                    {"x": 100, "y": 100, "width": 200, "height": 200, "confidence": 0.95}
                ]
            },
            "created_at": self.fake.date_time_this_year().isoformat(),
        }
        result.update(overrides)
        return result
    
    def create_test_image(
        self, 
        width: int = 800, 
        height: int = 600, 
        color: str = None,
        text: str = None,
        format: str = 'JPEG',
        save_path: Optional[Path] = None
    ) -> Path:
        """Generate a test image file"""
        if not color:
            color = self.fake.hex_color()
            
        if not text:
            text = f"Test Image {self.fake.uuid4()[:8]}"
        
        img = Image.new('RGB', (width, height), color=color)
        draw = ImageDraw.Draw(img)
        
        # Add some text to make the image more interesting
        draw.text((10, 10), text, fill=(0, 0, 0))
        
        if not save_path:
            save_path = self.temp_dir / f"test_image_{self.fake.uuid4()}.{format.lower()}"
        
        img.save(save_path, format=format)
        return save_path
    
    def create_test_video(
        self, 
        duration: int = 5, 
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        save_path: Optional[Path] = None
    ) -> Path:
        """Generate a test video file"""
        if not save_path:
            save_path = self.temp_dir / f"test_video_{self.fake.uuid4()}.mp4"
        
        # Create a video with random frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        
        for _ in range(duration * fps):
            # Create a frame with random noise
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        return save_path
    
    def create_test_audio(
        self, 
        duration: int = 5,  # seconds
        sample_rate: int = 44100,
        save_path: Optional[Path] = None
    ) -> Path:
        """Generate a test audio file"""
        import wave
        import struct
        
        if not save_path:
            save_path = self.temp_dir / f"test_audio_{self.fake.uuid4()}.wav"
        
        n_frames = duration * sample_rate
        
        with wave.open(str(save_path), 'w') as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)   # 2 bytes per sample
            wf.setframerate(sample_rate)
            
            # Generate a simple sine wave
            for i in range(n_frames):
                # 440 Hz sine wave
                value = int(32767.0 * np.sin(2.0 * np.pi * 440.0 * i / sample_rate))
                data = struct.pack('<h', value)
                wf.writeframes(data)
        
        return save_path

# Global test data factory instance
test_data = TestDataFactory()

# Clean up on exit
import atexit
atexit.register(test_data.cleanup)
