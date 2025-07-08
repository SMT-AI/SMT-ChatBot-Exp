import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any, Optional
from transformers import BertModel, BertTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BertClassifierError(Exception):
    """Base exception class for BERT classifier errors"""
    pass

class ModelLoadError(BertClassifierError):
    """Raised when model loading fails"""
    pass

class PredictionError(BertClassifierError):
    """Raised when prediction fails"""
    pass

class CBv2Dataset(Dataset):
    """Dataset class for BERT intent classification"""
    
    def __init__(self, texts: List[str], main_intents: List[str], 
                 sub_intents: List[str], tokenizer: Any, max_len: int = 128):
        self.texts = texts
        self.main_intents = main_intents
        self.sub_intents = sub_intents
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx])
            main_intent = self.main_intents[idx]
            sub_intent = self.sub_intents[idx]
            
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'main_intent': torch.tensor(main_intent, dtype=torch.long),
                'sub_intent': torch.tensor(sub_intent, dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Error processing dataset item at index {idx}: {str(e)}")
            raise

class HierarchicalIntentClassifier(nn.Module):
    """Hierarchical BERT classifier for main and sub-intents"""
    
    def __init__(self, intent_structure: Dict[str, Dict[str, List[str]]]):
        super().__init__()
        try:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.drop = nn.Dropout(0.3)
            
            # Main intent classifier
            self.n_main_intents = len(intent_structure)
            self.main_classifier = nn.Linear(768, self.n_main_intents)
            
            # Create sub-classifiers for each main intent
            self.sub_classifiers = nn.ModuleDict({
                main_intent: nn.Linear(768, len(sub_intents))
                for main_intent, sub_intents in intent_structure.items()
            })
            
            # Store the structure for reference
            self.intent_structure = intent_structure
            
        except Exception as e:
            logger.error(f"Error initializing hierarchical classifier: {str(e)}")
            raise ModelLoadError(f"Failed to initialize model: {str(e)}")

    def forward(self, input_ids, attention_mask, main_intent=None):
        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            dropped = self.drop(pooled_output)
            
            # Get main intent prediction
            main_output = self.main_classifier(dropped)
            
            if main_intent is None:
                main_intent = torch.argmax(main_output, dim=1)
            
            # Get sub-intent prediction for the specific main intent
            batch_size = input_ids.size(0)
            sub_outputs = torch.zeros(
                batch_size, 
                max(len(subs) for subs in self.intent_structure.values())
            ).to(input_ids.device)
            
            for i in range(batch_size):
                intent_name = list(self.intent_structure.keys())[main_intent[i]]
                sub_classifier = self.sub_classifiers[intent_name]
                sub_output = sub_classifier(dropped[i].unsqueeze(0))
                sub_outputs[i, :sub_output.size(1)] = sub_output
                
            return main_output, sub_outputs
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise PredictionError(f"Forward pass failed: {str(e)}")

class IntentClassifier:
    """Main interface for intent classification"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.7):
        """
        Initialize the intent classifier.
        
        Args:
            model_path: Path to the saved model file
            confidence_threshold: Threshold for confidence score (0.0 to 1.0)
        """
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
            
        self.device = torch.device('cpu') #CPU only inference
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.intent_structure = {}
        self.main_intent_to_id = {}
        self.sub_intent_to_id = {}
        self.confidence_threshold = confidence_threshold
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        """
        Load the model from a saved checkpoint.
        
        Args:
            path: Path to the model checkpoint
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.intent_structure = checkpoint['intent_structure']
            self.model = HierarchicalIntentClassifier(self.intent_structure)
            self.model.load_state_dict(checkpoint['model_state'])
            self.main_intent_to_id = checkpoint['main_intent_to_id']
            self.sub_intent_to_id = checkpoint['sub_intent_to_id']
            self.model.to(self.device)
            logger.info(f"Model successfully loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            raise ModelLoadError(f"Failed to load model from {path}: {str(e)}")

    def set_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intents for given text.
        
        Args:
            text: Input text for prediction
            
        Returns:
            Dictionary containing:
                - status: 'success' or 'llm_fallback'
                - main_intent: Predicted main intent (if success)
                - sub_intent: Predicted sub-intent (if success)
                - confidence: Dictionary with confidence scores
                - error: Error message (if any)
        """
        if not text or not text.strip():
            return {
                'status': 'error',
                'error': 'Empty or invalid input text',
                'confidence': {'main': 0.0, 'sub': 0.0}
            }

        if not self.model:
            return {
                'status': 'error',
                'error': 'Model not loaded',
                'confidence': {'main': 0.0, 'sub': 0.0}
            }

        try:
            self.model.eval()
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                main_outputs, _ = self.model(input_ids, attention_mask)
                main_probs = torch.softmax(main_outputs, dim=1)
                main_confidence, main_predicted = torch.max(main_probs, 1)
                
                # Check confidence threshold
                if main_confidence.item() < self.confidence_threshold:
                    return {
                        'status': 'llm_fallback',
                        'confidence': {
                            'main': main_confidence.item(),
                            'sub': 0.0
                        },
                        'original_query': text
                    }
                
                # Get main intent prediction
                main_id_to_intent = {v: k for k, v in self.main_intent_to_id.items()}
                predicted_main_intent = main_id_to_intent[main_predicted.item()]
                
                # Get sub-intent prediction
                dropped = self.model.drop(self.model.bert(input_ids, attention_mask)[1])
                sub_classifier = self.model.sub_classifiers[predicted_main_intent]
                sub_outputs = sub_classifier(dropped)
                sub_probs = torch.softmax(sub_outputs, dim=1)
                sub_confidence, sub_predicted = torch.max(sub_probs, 1)
                
                sub_id_to_intent = {v: k for k, v in self.sub_intent_to_id[predicted_main_intent].items()}
                predicted_sub_intent = sub_id_to_intent[sub_predicted.item()]
                
                return {
                    'status': 'success',
                    'main_intent': predicted_main_intent,
                    'sub_intent': predicted_sub_intent,
                    'confidence': {
                        'main': main_confidence.item(),
                        'sub': sub_confidence.item()
                    }
                }
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'status': 'error',
                'error': f"Prediction failed: {str(e)}",
                'confidence': {'main': 0.0, 'sub': 0.0}
            }