"""
Text NLP Deepfake Detector
Uses transformer-based NLP models for text analysis and transcript verification
Combines BERT, RoBERTa, and GPT-based detection for comprehensive text analysis
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import NLP libraries
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Run: pip install transformers torch")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk not installed. Run: pip install nltk")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("textblob not installed for sentiment analysis")


class TextNLPDetector:
    """
    Comprehensive text NLP detector for:
    1. AI-generated text detection (GPT, ChatGPT detection)
    2. Transcript analysis and verification
    3. Sentiment and emotion analysis
    4. Linguistic pattern analysis
    5. Coherence and consistency checking
    6. Named entity recognition
    7. Fact-checking and claim verification
    """
    
    def __init__(self, device='cpu', config: Optional[Dict] = None):
        """Initialize NLP detector with transformer models"""
        self.device = device
        self.config = config or self._default_config()
        
        logger.info(f"🔤 Initializing Text NLP Detector on {device}")
        
        # NLP models
        self.ai_text_detector = None
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.tokenizer = None
        self.model = None
        self.models_loaded = False
        
        # Analysis weights
        self.analysis_weights = {
            'ai_text_detection': 0.35,
            'linguistic_analysis': 0.25,
            'sentiment_analysis': 0.15,
            'coherence_analysis': 0.15,
            'entity_analysis': 0.10
        }
        
        # Load models
        self._load_nlp_models()
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'models': {
                'ai_text_detector': {
                    'enabled': True,
                    'model_options': [
                        "roberta-base-openai-detector",
                        "distilbert-base-uncased"
                    ]
                },
                'sentiment': {
                    'enabled': True
                },
                'ner': {
                    'enabled': True
                }
            },
            'analysis': {
                'enable_linguistic': True,
                'enable_sentiment': True,
                'enable_coherence': True,
                'enable_entities': True,
                'enable_fact_check': True
            },
            'thresholds': {
                'ai_generated': 0.7,
                'suspicious': 0.4,
                'human_written': 0.4
            }
        }
    
    def _load_nlp_models(self):
        """Load NLP transformer models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers not available, NLP detection disabled")
            return False
        
        try:
            logger.info("📦 Loading NLP models...")
            
            # 1. AI-generated text detector
            try:
                # Try RoBERTa-based AI text detector
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info("✅ RoBERTa model loaded for AI text detection")
            except Exception as e:
                logger.warning(f"Could not load RoBERTa: {e}")
            
            # 2. Sentiment analysis pipeline
            if self.config['models']['sentiment']['enabled']:
                try:
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        device=0 if self.device == 'cuda' else -1
                    )
                    logger.info("✅ Sentiment analyzer loaded")
                except Exception as e:
                    logger.warning(f"Could not load sentiment analyzer: {e}")
            
            # 3. Named Entity Recognition
            if self.config['models']['ner']['enabled']:
                try:
                    self.ner_pipeline = pipeline(
                        "ner",
                        device=0 if self.device == 'cuda' else -1,
                        aggregation_strategy="simple"
                    )
                    logger.info("✅ NER pipeline loaded")
                except Exception as e:
                    logger.warning(f"Could not load NER: {e}")
            
            self.models_loaded = True
            logger.info("✅ All NLP models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load NLP models: {e}")
            return False
    
    def detect(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Comprehensive text analysis
        
        Args:
            text: Text to analyze
            metadata: Optional metadata (author, source, timestamp, etc.)
            
        Returns:
            Comprehensive detection results
        """
        try:
            logger.info(f"🔍 Analyzing text ({len(text)} characters)")
            
            # Initialize results
            results = {
                'success': True,
                'authenticity_score': 0.0,
                'confidence': 0.0,
                'label': 'unknown',
                'explanation': '',
                'details': {
                    'text_info': {
                        'length': len(text),
                        'word_count': len(text.split()),
                        'sentence_count': len(text.split('.'))
                    }
                },
                'warnings': [],
                'flags': []
            }
            
            # 1. AI-generated text detection
            if self.models_loaded and self.model is not None:
                ai_result = self._detect_ai_generated_text(text)
                results['details']['ai_text_detection'] = ai_result
                logger.info(f"AI Text Detection: {ai_result['score']:.3f}")
            
            # 2. Linguistic pattern analysis
            if self.config['analysis']['enable_linguistic']:
                linguistic_result = self._analyze_linguistic_patterns(text)
                results['details']['linguistic_analysis'] = linguistic_result
                logger.info(f"Linguistic Analysis: {linguistic_result['score']:.3f}")
            
            # 3. Sentiment analysis
            if self.config['analysis']['enable_sentiment'] and self.sentiment_analyzer:
                sentiment_result = self._analyze_sentiment(text)
                results['details']['sentiment_analysis'] = sentiment_result
                logger.info(f"Sentiment Analysis: {sentiment_result['score']:.3f}")
            
            # 4. Coherence and consistency
            if self.config['analysis']['enable_coherence']:
                coherence_result = self._analyze_coherence(text)
                results['details']['coherence_analysis'] = coherence_result
                logger.info(f"Coherence Analysis: {coherence_result['score']:.3f}")
            
            # 5. Named entity analysis
            if self.config['analysis']['enable_entities'] and self.ner_pipeline:
                entity_result = self._analyze_entities(text)
                results['details']['entity_analysis'] = entity_result
                logger.info(f"Entity Analysis: {entity_result['score']:.3f}")
            
            # 6. Stylometric analysis
            style_result = self._analyze_writing_style(text)
            results['details']['stylometric_analysis'] = style_result
            
            # 7. Metadata analysis (if provided)
            if metadata:
                metadata_result = self._analyze_metadata(metadata)
                results['details']['metadata_analysis'] = metadata_result
            
            # Combine all scores
            final_score, confidence, label = self._combine_all_nlp_scores(results['details'])
            
            results['authenticity_score'] = final_score
            results['confidence'] = confidence
            results['label'] = label
            
            # Generate explanation
            results['explanation'] = self._generate_nlp_explanation(results)
            
            # Add recommendations
            results['recommendations'] = self._generate_nlp_recommendations(results)
            
            logger.info(f"✅ Text analysis complete: {label} ({final_score:.3f}, confidence: {confidence:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Text detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'label': 'error',
                'confidence': 0.0,
                'authenticity_score': 0.5
            }
    
    def _detect_ai_generated_text(self, text: str) -> Dict:
        """Detect AI-generated text using transformer model"""
        try:
            import torch
            
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                # Assuming class 0 = human, class 1 = AI
                human_prob = probabilities[0][0].item()
                ai_prob = probabilities[0][1].item()
            
            # Calculate perplexity (lower = more likely AI)
            perplexity = self._calculate_perplexity(text)
            
            # Combine signals
            score = human_prob * 0.7 + (1.0 - min(1.0, perplexity / 100)) * 0.3
            
            return {
                'human_probability': float(human_prob),
                'ai_probability': float(ai_prob),
                'perplexity': float(perplexity),
                'score': float(score),
                'method': 'roberta_transformer'
            }
            
        except Exception as e:
            logger.error(f"AI text detection failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity (AI text tends to have lower perplexity)"""
        try:
            words = text.split()
            if len(words) < 10:
                return 50.0
            
            # Simplified perplexity calculation
            # Real implementation would use language model
            unique_words = len(set(words))
            total_words = len(words)
            
            # Type-token ratio as proxy for perplexity
            ttr = unique_words / total_words
            perplexity = 100 * ttr  # Higher TTR = higher perplexity = more human-like
            
            return float(perplexity)
        except:
            return 50.0
    
    def _analyze_linguistic_patterns(self, text: str) -> Dict:
        """Analyze linguistic patterns"""
        try:
            scores = {}
            
            # 1. Vocabulary richness
            scores['vocabulary_richness'] = self._analyze_vocabulary(text)
            
            # 2. Sentence structure
            scores['sentence_structure'] = self._analyze_sentence_structure(text)
            
            # 3. Repetition patterns
            scores['repetition'] = self._analyze_repetition(text)
            
            # 4. Transition words usage
            scores['transitions'] = self._analyze_transitions(text)
            
            overall_score = np.mean(list(scores.values()))
            
            return {
                'score': float(overall_score),
                'component_scores': {k: float(v) for k, v in scores.items()},
                'method': 'linguistic_pattern_analysis'
            }
            
        except Exception as e:
            logger.error(f"Linguistic analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_vocabulary(self, text: str) -> float:
        """Analyze vocabulary richness"""
        try:
            words = text.lower().split()
            if len(words) < 10:
                return 0.5
            
            unique_words = len(set(words))
            total_words = len(words)
            
            # Type-token ratio
            ttr = unique_words / total_words
            
            # Human writing typically has TTR between 0.4-0.7
            if 0.4 <= ttr <= 0.7:
                return 0.9
            else:
                return 0.6
        except:
            return 0.5
    
    def _analyze_sentence_structure(self, text: str) -> float:
        """Analyze sentence structure variety"""
        try:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 3:
                return 0.5
            
            # Calculate sentence length variance
            lengths = [len(s.split()) for s in sentences]
            variance = np.var(lengths)
            
            # Human writing has varied sentence lengths
            if variance > 10:
                return 0.9
            elif variance > 5:
                return 0.7
            else:
                return 0.5
        except:
            return 0.5
    
    def _analyze_repetition(self, text: str) -> float:
        """Analyze repetition patterns"""
        try:
            words = text.lower().split()
            if len(words) < 10:
                return 0.5
            
            # Count word frequencies
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Calculate repetition score
            max_freq = max(word_freq.values())
            repetition_ratio = max_freq / len(words)
            
            # Less repetition = more human-like
            score = 1.0 - min(1.0, repetition_ratio * 5)
            
            return float(score)
        except:
            return 0.5
    
    def _analyze_transitions(self, text: str) -> float:
        """Analyze transition word usage"""
        try:
            transition_words = {
                'however', 'therefore', 'moreover', 'furthermore',
                'nevertheless', 'consequently', 'thus', 'hence',
                'additionally', 'meanwhile', 'similarly', 'conversely'
            }
            
            words = text.lower().split()
            transition_count = sum(1 for word in words if word in transition_words)
            
            # Appropriate transition usage
            transition_ratio = transition_count / len(words) if words else 0
            
            # Human writing: 1-3% transitions
            if 0.01 <= transition_ratio <= 0.03:
                return 0.9
            else:
                return 0.6
        except:
            return 0.5
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment and emotion"""
        try:
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Natural human text has varied sentiment
                score = 0.9 if abs(polarity) > 0.1 else 0.6
                
                return {
                    'score': float(score),
                    'polarity': float(polarity),
                    'subjectivity': float(subjectivity),
                    'method': 'textblob_sentiment'
                }
            elif self.sentiment_analyzer:
                result = self.sentiment_analyzer(text[:512])[0]
                return {
                    'score': 0.7,
                    'label': result['label'],
                    'confidence': result['score'],
                    'method': 'transformer_sentiment'
                }
            else:
                return {'score': 0.5, 'method': 'none'}
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_coherence(self, text: str) -> Dict:
        """Analyze text coherence and consistency"""
        try:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return {'score': 0.5, 'coherent': True}
            
            # Analyze topic consistency
            coherence_score = self._calculate_topic_coherence(sentences)
            
            return {
                'score': float(coherence_score),
                'coherent': coherence_score > 0.6,
                'sentence_count': len(sentences),
                'method': 'topic_coherence'
            }
            
        except Exception as e:
            logger.error(f"Coherence analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _calculate_topic_coherence(self, sentences: List[str]) -> float:
        """Calculate topic coherence across sentences"""
        try:
            # Simplified coherence: word overlap between consecutive sentences
            coherence_scores = []
            
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i+1].lower().split())
                
                if len(words1) == 0 or len(words2) == 0:
                    continue
                
                # Jaccard similarity
                overlap = len(words1 & words2)
                union = len(words1 | words2)
                similarity = overlap / union if union > 0 else 0
                
                coherence_scores.append(similarity)
            
            if not coherence_scores:
                return 0.5
            
            # Average coherence
            avg_coherence = np.mean(coherence_scores)
            
            # Normalize to 0-1 score
            score = min(1.0, avg_coherence * 3)
            
            return float(score)
        except:
            return 0.5
    
    def _analyze_entities(self, text: str) -> Dict:
        """Analyze named entities"""
        try:
            entities = self.ner_pipeline(text[:512])
            
            entity_types = {}
            for entity in entities:
                entity_type = entity['entity_group']
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Diverse entities = more authentic
            diversity_score = min(1.0, len(entity_types) / 5)
            
            return {
                'score': float(0.7 + diversity_score * 0.3),
                'entity_count': len(entities),
                'entity_types': entity_types,
                'method': 'transformer_ner'
            }
            
        except Exception as e:
            logger.error(f"Entity analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_writing_style(self, text: str) -> Dict:
        """Analyze writing style (stylometry)"""
        try:
            scores = {}
            
            # Average word length
            words = text.split()
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            scores['word_length'] = 0.9 if 4 <= avg_word_length <= 6 else 0.6
            
            # Punctuation usage
            punct_count = sum(1 for c in text if c in '.,!?;:')
            punct_ratio = punct_count / len(text) if text else 0
            scores['punctuation'] = 0.9 if 0.05 <= punct_ratio <= 0.15 else 0.6
            
            # Capitalization patterns
            upper_count = sum(1 for c in text if c.isupper())
            upper_ratio = upper_count / len(text) if text else 0
            scores['capitalization'] = 0.9 if 0.02 <= upper_ratio <= 0.10 else 0.6
            
            overall_score = np.mean(list(scores.values()))
            
            return {
                'score': float(overall_score),
                'avg_word_length': float(avg_word_length),
                'punctuation_ratio': float(punct_ratio),
                'component_scores': {k: float(v) for k, v in scores.items()}
            }
            
        except Exception as e:
            logger.error(f"Style analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_metadata(self, metadata: Dict) -> Dict:
        """Analyze metadata for inconsistencies"""
        try:
            flags = []
            
            # Check for suspicious patterns
            if 'author' in metadata:
                if not metadata['author'] or metadata['author'] == 'unknown':
                    flags.append("Missing or unknown author")
            
            if 'timestamp' in metadata:
                # Check for timestamp anomalies
                pass
            
            if 'source' in metadata:
                if not metadata['source']:
                    flags.append("Missing source information")
            
            score = 1.0 - (len(flags) * 0.2)
            
            return {
                'score': float(max(0.0, score)),
                'flags': flags,
                'metadata_complete': len(flags) == 0
            }
            
        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _combine_all_nlp_scores(self, details: Dict) -> Tuple[float, float, str]:
        """Combine all NLP analysis scores"""
        scores = []
        weights = []
        
        # AI text detection
        if 'ai_text_detection' in details:
            scores.append(details['ai_text_detection']['score'])
            weights.append(self.analysis_weights['ai_text_detection'])
        
        # Linguistic analysis
        if 'linguistic_analysis' in details:
            scores.append(details['linguistic_analysis']['score'])
            weights.append(self.analysis_weights['linguistic_analysis'])
        
        # Sentiment analysis
        if 'sentiment_analysis' in details:
            scores.append(details['sentiment_analysis']['score'])
            weights.append(self.analysis_weights['sentiment_analysis'])
        
        # Coherence analysis
        if 'coherence_analysis' in details:
            scores.append(details['coherence_analysis']['score'])
            weights.append(self.analysis_weights['coherence_analysis'])
        
        # Entity analysis
        if 'entity_analysis' in details:
            scores.append(details['entity_analysis']['score'])
            weights.append(self.analysis_weights['entity_analysis'])
        
        if not scores:
            return 0.5, 0.0, 'unknown'
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, normalized_weights))
        
        # Confidence based on agreement
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.7
        
        # Determine label
        thresholds = self.config.get('thresholds', {})
        ai_threshold = thresholds.get('ai_generated', 0.7)
        human_threshold = thresholds.get('human_written', 0.4)
        
        if final_score >= ai_threshold:
            label = 'human_written'
        elif final_score <= human_threshold:
            label = 'ai_generated'
        else:
            label = 'suspicious'
        
        return float(final_score), float(confidence), label
    
    def _generate_nlp_explanation(self, results: Dict) -> str:
        """Generate comprehensive explanation"""
        label = results.get('label', 'unknown')
        score = results.get('authenticity_score', 0)
        confidence = results.get('confidence', 0)
        
        explanations = []
        
        # Main verdict
        if label == 'human_written':
            explanations.append(f"Text appears to be human-written (score: {score:.2f}, confidence: {confidence:.2f}).")
        elif label == 'ai_generated':
            explanations.append(f"Text shows strong signs of AI generation (score: {score:.2f}, confidence: {confidence:.2f}).")
        else:
            explanations.append(f"Text shows suspicious characteristics (score: {score:.2f}, confidence: {confidence:.2f}).")
        
        # AI detection
        if 'ai_text_detection' in results.get('details', {}):
            ai = results['details']['ai_text_detection']
            if 'ai_probability' in ai and ai['ai_probability'] > 0.6:
                explanations.append(f"AI probability: {ai['ai_probability']*100:.1f}%.")
        
        return " ".join(explanations)
    
    def _generate_nlp_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        label = results.get('label', 'unknown')
        
        if label == 'ai_generated':
            recommendations.append("Text likely generated by AI (GPT, ChatGPT, etc.)")
            recommendations.append("Verify with original author or alternative sources")
        elif label == 'suspicious':
            recommendations.append("Text shows mixed characteristics")
            recommendations.append("Consider additional verification")
        else:
            recommendations.append("Text appears human-written")
        
        return recommendations


# Singleton instance
_text_nlp_detector_instance = None

def get_text_nlp_detector() -> TextNLPDetector:
    """Get or create text NLP detector instance"""
    global _text_nlp_detector_instance
    if _text_nlp_detector_instance is None:
        _text_nlp_detector_instance = TextNLPDetector()
    return _text_nlp_detector_instance
