"""
Text Deepfake Detection using HuggingFace Transformers
Implements advanced text analysis for detecting AI-generated content, fake news, and manipulated text.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# HuggingFace Transformers
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModel, AutoConfig,
        RobertaModel, RobertaTokenizer, RobertaForSequenceClassification,
        BertModel, BertTokenizer, BertForSequenceClassification,
        DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification,
        GPT2Model, GPT2Tokenizer,
        T5Model, T5Tokenizer, T5ForConditionalGeneration,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available for text analysis")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDeepfakeDetector(nn.Module):
    """Advanced text deepfake detection using multiple transformer models."""
    
    def __init__(
        self,
        model_names: List[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_ensemble: bool = True
    ):
        super().__init__()
        
        self.device = device
        self.use_ensemble = use_ensemble
        
        # Default models for text deepfake detection
        if model_names is None:
            model_names = [
                "roberta-base-openai-detector",  # OpenAI content detector
                "bert-base-uncased",  # General purpose
                "distilbert-base-uncased",  # Lightweight
                "microsoft/DialoGPT-medium",  # Conversation analysis
                "facebook/bart-large-mnli"  # Natural language inference
            ]
        
        self.models = {}
        self.tokenizers = {}
        self.model_weights = [0.4, 0.3, 0.2, 0.05, 0.05]  # Ensemble weights
        
        # Load models
        if TRANSFORMERS_AVAILABLE:
            self._load_models(model_names)
        else:
            logger.error("Transformers not available. Cannot load text models.")
        
        self.to(device)
        
    def _load_models(self, model_names: List[str]):
        """Load multiple transformer models for ensemble detection."""
        for i, model_name in enumerate(model_names):
            try:
                logger.info(f"Loading text model: {model_name}")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.tokenizers[model_name] = tokenizer
                
                # Load model
                if "openai-detector" in model_name:
                    # Special handling for OpenAI detector
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                elif "bart-large-mnli" in model_name:
                    # For NLI tasks
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                elif "DialoGPT" in model_name:
                    # For generative text analysis
                    model = AutoModel.from_pretrained(model_name)
                else:
                    # General sequence classification
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        num_labels=2  # Binary classification: real vs fake
                    )
                
                model.eval()
                model.to(self.device)
                self.models[model_name] = model
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                continue
    
    def preprocess_text(self, text: str, model_name: str) -> Dict[str, Any]:
        """Preprocess text for specific model."""
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return inputs
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for deepfake detection."""
        if not TRANSFORMERS_AVAILABLE or not self.models:
            return {
                "error": "No transformer models available",
                "is_fake": None,
                "confidence": 0.0
            }
        
        results = {}
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                # Preprocess
                inputs = self.preprocess_text(text, model_name)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                with torch.no_grad():
                    if "DialoGPT" in model_name:
                        # Special handling for generative models
                        outputs = model(**inputs, output_hidden_states=True)
                        # Use hidden states for analysis
                        hidden_states = outputs.last_hidden_state
                        # Simple classification based on hidden state patterns
                        fake_score = torch.mean(torch.abs(hidden_states)) * 0.1
                        logits = torch.tensor([[0.0, fake_score]])
                    else:
                        outputs = model(**inputs)
                        logits = outputs.logits
                
                # Get probabilities
                probs = F.softmax(logits, dim=-1)
                fake_prob = probs[0][1].item() if probs.shape[1] > 1 else 0.0
                
                results[model_name] = {
                    "fake_probability": fake_prob,
                    "real_probability": 1.0 - fake_prob,
                    "confidence": max(fake_prob, 1.0 - fake_prob)
                }
                
                predictions.append(fake_prob)
                
            except Exception as e:
                logger.error(f"Error analyzing with {model_name}: {e}")
                continue
        
        # Ensemble prediction
        if predictions and self.use_ensemble:
            # Weighted average
            weighted_prediction = sum(p * w for p, w in zip(predictions, self.model_weights[:len(predictions)]))
            final_fake_prob = weighted_prediction
        elif predictions:
            final_fake_prob = np.mean(predictions)
        else:
            final_fake_prob = 0.5
        
        return {
            "is_fake": final_fake_prob > 0.5,
            "fake_probability": final_fake_prob,
            "real_probability": 1.0 - final_fake_prob,
            "confidence": max(final_fake_prob, 1.0 - final_fake_prob),
            "model_results": results,
            "ensemble_used": self.use_ensemble
        }
    
    def analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns that indicate AI generation."""
        patterns = {
            "repetition_score": self._calculate_repetition(text),
            "perplexity_score": self._calculate_perplexity(text),
            "burstiness_score": self._calculate_burstiness(text),
            "vocabulary_richness": self._calculate_vocabulary_richness(text),
            "sentence_structure": self._analyze_sentence_structure(text),
            "emotional_neutrality": self._analyze_emotional_neutrality(text)
        }
        
        return patterns
    
    def _calculate_repetition(self, text: str) -> float:
        """Calculate text repetition score."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        unique_words = set(words)
        repetition = 1.0 - (len(unique_words) / len(words))
        return repetition
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity (simplified)."""
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        # Simplified perplexity based on word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate average inverse frequency
        total_freq = sum(word_freq.values())
        avg_freq = total_freq / len(word_freq)
        perplexity = len(word_freq) / avg_freq if avg_freq > 0 else 0.0
        
        return min(perplexity / 100.0, 1.0)  # Normalize to 0-1
    
    def _calculate_burstiness(self, text: str) -> float:
        """Calculate burstiness (variance in word length)."""
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        word_lengths = [len(word) for word in words]
        avg_length = np.mean(word_lengths)
        variance = np.var(word_lengths)
        
        # Normalize variance
        burstiness = min(variance / (avg_length ** 2) if avg_length > 0 else 1.0, 1.0)
        return burstiness
    
    def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calculate vocabulary richness."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        unique_words = set(words)
        richness = len(unique_words) / len(words)
        return richness
    
    def _analyze_sentence_structure(self, text: str) -> Dict[str, float]:
        """Analyze sentence structure patterns."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return {"avg_length": 0.0, "length_variance": 0.0}
        
        sentence_lengths = [len(s.strip().split()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return {"avg_length": 0.0, "length_variance": 0.0}
        
        avg_length = np.mean(sentence_lengths)
        length_variance = np.var(sentence_lengths)
        
        return {
            "avg_length": avg_length / 50.0,  # Normalize
            "length_variance": min(length_variance / 100.0, 1.0)
        }
    
    def _analyze_emotional_neutrality(self, text: str) -> float:
        """Analyze emotional neutrality (simplified)."""
        # Simple heuristic based on emotional words
        emotional_words = {
            'happy', 'sad', 'angry', 'excited', 'worried', 'scared', 'love', 'hate',
            'amazing', 'terrible', 'wonderful', 'awful', 'fantastic', 'horrible'
        }
        
        words = text.lower().split()
        emotional_count = sum(1 for word in words if word in emotional_words)
        
        neutrality = 1.0 - (emotional_count / len(words)) if words else 1.0
        return neutrality


# Convenience function for quick analysis
def analyze_text_deepfake(text: str) -> Dict[str, Any]:
    """Quick text deepfake analysis."""
    detector = TextDeepfakeDetector()
    return detector.analyze_text(text)
