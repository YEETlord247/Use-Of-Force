import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import (
    BertTokenizer, BertForSequenceClassification, BertModel, 
    TrainingArguments, Trainer, EarlyStoppingCallback,
    PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer, AutoConfig,
    PretrainedConfig, TrainerState
)
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import re
import evaluate
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import multiprocessing
import os
import psutil
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
import spacy
from collections import Counter
import random
import logging
import sys
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import traceback

@dataclass
class ForceClassificationOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    force_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Convert the output into a tuple and apply the slice
            outputs = []
            if self.loss is not None:
                outputs.append(self.loss)
            if self.logits is not None:
                outputs.append(self.logits)
            if self.force_logits is not None:
                outputs.append(self.force_logits)
            if self.hidden_states is not None:
                outputs.append(self.hidden_states)
            if self.attentions is not None:
                outputs.append(self.attentions)
            return tuple(outputs[key])
        elif isinstance(key, int):
            if key == 0:
                return self.loss
            elif key == 1:
                return self.logits
            elif key == 2:
                return self.force_logits
            else:
                raise IndexError(f"Index {key} is out of bounds")
        elif isinstance(key, str):
            return getattr(self, key)
        raise TypeError(f"Invalid key type: {type(key)}")

    def __iter__(self):
        yield self.loss
        yield self.logits
        yield self.force_logits
        if self.hidden_states is not None:
            yield self.hidden_states
        if self.attentions is not None:
            yield self.attentions

    def to(self, device):
        """Move the output to a specific device."""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
            elif isinstance(value, tuple) and all(isinstance(x, torch.Tensor) for x in value):
                setattr(self, key, tuple(x.to(device) for x in value))
        return self

class ForceClassificationConfig(PretrainedConfig):
    model_type = "force_classification"
    
    def __init__(
        self,
        num_labels=6,
        hidden_size=768,
        num_attention_heads=8,
        feature_fusion_size=64,
        dropout_rate=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.feature_fusion_size = feature_fusion_size
        self.dropout_rate = dropout_rate

class ImprovedForceClassificationEnsemble(PreTrainedModel):
    config_class = ForceClassificationConfig
    base_model_prefix = "force_classifier"
    supports_gradient_checkpointing = True
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, config.feature_fusion_size)
        )
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads
        )
        
        # Binary force detector
        self.force_detector = nn.Linear(config.hidden_size, 2)
        
        # Main classifier (updated for 5 classes)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size + config.feature_fusion_size, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 5)  # Changed to 5 classes
        )
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        force_indicators=None,
        has_weapon=None,
        has_resistance=None,
        has_compliance=None,
        force_level_hint=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get the [CLS] token embedding
        pooled_output = outputs[0][:, 0]  # Shape: [batch_size, hidden_size]
        
        # Process additional features if provided
        if all(f is not None for f in [force_indicators, has_weapon, has_resistance, has_compliance, force_level_hint]):
            additional_features = torch.stack([
                force_indicators,
                has_weapon,
                has_resistance,
                has_compliance,
                force_level_hint
            ], dim=1).float()
            
            # Fuse additional features
            fused_features = self.feature_fusion(additional_features)
            
            # Concatenate with BERT output
            combined_features = torch.cat([pooled_output, fused_features], dim=1)
        else:
            # If features are not provided, use zero tensor
            device = pooled_output.device
            fused_features = torch.zeros(pooled_output.size(0), self.config.feature_fusion_size, device=device)
            combined_features = torch.cat([pooled_output, fused_features], dim=1)
        
        # Get logits for 5 classes
        logits = self.classifier(combined_features)  # Shape: [batch_size, 5]
        
        # Binary force detection
        force_logits = self.force_detector(pooled_output)  # Shape: [batch_size, 2]
        force_logits = force_logits[:, 1]  # Take positive class logits only
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Adjust labels to be 0-based
            adjusted_labels = labels - 1
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, adjusted_labels)
            
            # Add binary force detection loss
            binary_labels = (labels > 0).float()
            force_loss = F.binary_cross_entropy_with_logits(force_logits, binary_labels)
            loss = loss + 0.3 * force_loss
        
        if not return_dict:
            output = (logits, force_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return ForceClassificationOutput(
            loss=loss,
            logits=logits,
            force_logits=force_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class CurriculumScheduler:
    def __init__(self, dataset, num_epochs=10):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.complexities = self._compute_complexities()

    def _compute_complexities(self):
        """Compute complexity scores for each example in the dataset."""
        complexities = []
        
        # Get the input_ids from the dataset
        if isinstance(self.dataset, torch.utils.data.Subset):
            # For Subset, we need to index into the underlying dataset
            dataset_items = [self.dataset.dataset[self.dataset.indices[i]] for i in range(len(self.dataset))]
        else:
            # For regular dataset, just get the items directly
            dataset_items = [self.dataset[i] for i in range(len(self.dataset))]
            
        # Compute complexity based on sequence length and other features
        for item in dataset_items:
            # Handle both dictionary and tuple formats
            if isinstance(item, dict):
                ids = item['input_ids']
            else:
                ids = item[0]  # Fallback for tuple format
                
            # Basic complexity based on sequence length (excluding padding)
            seq_length = torch.sum(ids != 0).item()  # Count non-padding tokens
            complexity = seq_length / 512.0  # Normalize by max sequence length
            
            # Add to complexities list
            complexities.append(complexity)
            
        return torch.tensor(complexities)

    def sample_batch(self, difficulty, dataset):
        """Sample a batch of examples based on current difficulty."""
        # Convert difficulty to a cutoff percentile
        percentile = 1.0 - difficulty
        cutoff_idx = max(1, int(len(dataset) * percentile))
        
        # Get sorted indices based on complexity
        sorted_indices = torch.argsort(self.complexities).tolist()
        
        # Select indices up to the cutoff
        if isinstance(dataset, torch.utils.data.Subset):
            selected_indices = [dataset.indices[i] for i in sorted_indices[:cutoff_idx]]
            return torch.utils.data.Subset(dataset.dataset, selected_indices)
        else:
            selected_indices = sorted_indices[:cutoff_idx]
            return torch.utils.data.Subset(dataset, selected_indices)

    def update_difficulty(self, epoch):
        """Update the difficulty based on the current epoch."""
        # Handle None epoch value at the start of training
        if epoch is None:
            epoch = 0
        self.current_epoch = epoch
        return min(1.0, (epoch + 1) / self.num_epochs)

class HierarchicalLoss(nn.Module):
    def __init__(self, num_labels=6, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, labels, force_logits=None):
        # Standard classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        if force_logits is not None:
            # Binary force detection loss
            force_labels = (labels > 0).float()
            force_loss = F.binary_cross_entropy_with_logits(
                force_logits.squeeze(), force_labels
            )
            return ce_loss + self.alpha * force_loss
        
        return ce_loss

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable Intel MKL/oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
torch.set_num_threads(psutil.cpu_count(logical=False))  # Use physical cores

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load spaCy model for advanced NLP
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class FocalLoss(nn.Module):
    def __init__(self, gamma=4.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, input, target):
        input = input.to(target.device)
        if self.weight is not None:
            self.weight = self.weight.to(target.device)
        ce_loss = nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

@dataclass
class TrainingMetrics:
    train_losses: list = field(default_factory=list)
    eval_losses: list = field(default_factory=list)
    eval_accuracies: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)

metrics_history = TrainingMetrics()

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

class CustomTrainer(Trainer):
    def __init__(self, *args, curriculum_scheduler=None, **kwargs):
        # Store curriculum_scheduler before calling parent's init
        self._curriculum_scheduler = curriculum_scheduler
        # Remove curriculum_scheduler from kwargs to avoid error
        if 'curriculum_scheduler' in kwargs:
            del kwargs['curriculum_scheduler']
        super().__init__(*args, **kwargs)
        
        # Initialize state
        if not hasattr(self, 'state'):
            self.state = TrainerState()
        if self.state.epoch is None:
            self.state.epoch = 0

    @property
    def curriculum_scheduler(self):
        return self._curriculum_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss with special handling for critical force levels.
        """
        # Ensure inputs is a dictionary
        if not isinstance(inputs, dict):
            inputs = dict(zip(['input_ids', 'attention_mask', 'labels'], inputs))

        # Move inputs to the model's device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Get labels before forward pass
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Ensure outputs are on the correct device
        outputs = outputs.to(device)
        
        # Compute loss
        loss = None
        if labels is not None:
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # Get class weights for weighted loss
                class_weights = self.compute_class_weights().to(device)
                
                # Apply focal loss with increased gamma for critical levels
                gamma_weights = torch.ones(5, device=device) * 4.0  # Base gamma for 5 classes
                gamma_weights[3] = 5.0  # Higher gamma for less lethal (index 3 for level 4)
                gamma_weights[4] = 6.0  # Highest gamma for lethal (index 4 for level 5)
                
                # Custom loss calculation
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)
                
                # Calculate focal weights - handle batch dimension properly
                targets = F.one_hot(labels - 1, num_classes=5).float()  # Subtract 1 from labels since we start from 1
                batch_gamma = gamma_weights[labels - 1].unsqueeze(1).expand(-1, 5)  # Expand to match probs shape
                focal_weights = (1 - probs) ** batch_gamma
                
                # Calculate weighted loss with proper broadcasting
                per_sample_loss = -(targets * log_probs) * focal_weights
                per_sample_loss = per_sample_loss * class_weights.unsqueeze(0)  # Broadcast class weights
                loss = per_sample_loss.mean()
                
                # Add force detection loss if available
                if outputs.force_logits is not None:
                    force_logits = outputs.force_logits.view(-1)
                    binary_labels = (labels > 0).float()
                    
                    # Higher weight for critical levels in binary detection
                    critical_mask = (labels >= 4).float()
                    force_weights = torch.ones_like(binary_labels) + critical_mask * 2.0
                    
                    force_loss = F.binary_cross_entropy_with_logits(
                        force_logits,
                        binary_labels,
                        weight=force_weights,
                        pos_weight=torch.tensor([3.0]).to(device)
                    )
                    loss = loss + 0.5 * force_loss

        return (loss, outputs) if return_outputs else loss

    def compute_class_weights(self):
        """
        Compute class weights based on the training dataset with stronger emphasis on critical force levels.
        """
        if not hasattr(self, '_class_weights'):
            # Get labels from the dataset
            if isinstance(self.train_dataset, torch.utils.data.Subset):
                labels = torch.tensor([self.train_dataset.dataset[idx]['labels'] for idx in self.train_dataset.indices])
            else:
                labels = torch.tensor([item['labels'] for item in self.train_dataset])
            
            # Compute class counts for classes 1-5 only
            class_counts = torch.zeros(5)  # Initialize counts for 5 classes
            for i in range(1, 6):  # Count occurrences of classes 1-5
                class_counts[i-1] = (labels == i).sum()
            
            total_samples = len(labels)
            
            # Compute inverse frequency weights
            weights = total_samples / (5 * class_counts.float())  # Use 5 for number of classes
            
            # Apply critical level boosting (for 5 classes)
            boost_factors = torch.tensor([1.0, 1.0, 1.5, 2.0, 2.5])  # Boost factors for levels 1-5
            weights = weights * boost_factors
            
            # Additional boost for critical levels
            critical_boost = torch.ones_like(weights)
            critical_boost[3] = 2.0  # Double importance for less lethal (level 4)
            critical_boost[4] = 2.5  # 2.5x importance for lethal (level 5)
            weights = weights * critical_boost
            
            # Normalize weights
            weights = weights / weights.sum() * len(weights)
            
            print("\nClass weights:")
            for i, w in enumerate(weights):
                print(f"Class {i+1}: {w:.4f}")  # i+1 because we start from level 1
            
            # Store the computed weights
            self._class_weights = weights
        
        return self._class_weights

    def get_train_dataloader(self):
        """
        Returns the training dataloader with curriculum learning applied.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        
        # Apply curriculum learning if scheduler is available
        if self.curriculum_scheduler is not None:
            difficulty = self.curriculum_scheduler.update_difficulty(self.state.epoch)
            train_dataset = self.curriculum_scheduler.sample_batch(difficulty, train_dataset)

        data_collator = self.data_collator
        if data_collator is None:
            data_collator = ForceDataCollator(self.tokenizer)

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

for r in range(100):

    def log(self, logs: Dict[str, float], iterator=None) -> None:
        """Collect metrics after each logging step"""
        super().log(logs)
        if "loss" in logs:
            metrics_history.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            metrics_history.eval_losses.append(logs["eval_loss"])
        if "eval_accuracy" in logs:
            metrics_history.eval_accuracies.append(logs["eval_accuracy"])
        if "learning_rate" in logs:
            metrics_history.learning_rates.append(logs["learning_rate"])

def plot_training_metrics():
    plt.figure(figsize=(15, 5))


    # Plot accuracies
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history.train_losses, label='Train Loss')
    plt.plot(metrics_history.eval_losses, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history.learning_rates, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Handle different output formats
    if isinstance(predictions, tuple):
        # If predictions is a tuple, get the main logits
        logits = predictions[0]
    else:
        logits = predictions
    
    # Ensure predictions are 2D
    if len(logits.shape) == 1:
        # If 1D, reshape to 2D
        logits = logits.reshape(-1, 1)
    
    # Get predicted classes
    if logits.shape[1] == 1:
        # Binary classification case
        predicted_classes = (logits > 0).astype(np.int64)
    else:
        # Multi-class classification case
        predicted_classes = np.argmax(logits, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted_classes, average='weighted')
    acc = accuracy_score(labels, predicted_classes)
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for i in range(1, 6):  # Changed to range 1-5
        mask = labels == i
        if mask.sum() > 0:
            class_accuracies[f'class_{i}_acc'] = accuracy_score(
                labels[mask], 
                predicted_classes[mask]
            )
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(labels, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Combine all metrics
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        **class_accuracies
    }
    
    return metrics

def preprocess_text(text):
    """Preprocess text to improve model robustness."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove punctuation except for basic sentence structure
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Normalize common variations
    text = re.sub(r'\b(please|sir|ma\'am|mam)\b', '', text)
    text = re.sub(r'\b(ok|okay|yeah|yes)\b', '', text)
    
    # Remove filler words
    text = re.sub(r'\b(um|uh|like|you know|i mean)\b', '', text)
    
    return text.strip()

def get_force_terms(force_level: int) -> dict:
    """Get force-specific terms and their importance weights for each force level."""
    force_terms = {
        0: {  # No force
            'comply': 0.8, 'cooperate': 0.8, 'calm': 0.9, 'peaceful': 0.9,
            'verbal': 0.7, 'talk': 0.6, 'discuss': 0.6
        },
        1: {  # Verbal commands
            'command': 0.8, 'order': 0.8, 'instruct': 0.7, 'direct': 0.7,
            'warn': 0.9, 'advise': 0.6, 'request': 0.5
        },
        2: {  # Soft physical
            'control': 0.8, 'restrain': 0.9, 'hold': 0.7, 'guide': 0.6,
            'escort': 0.7, 'position': 0.6, 'stance': 0.5
        },
        3: {  # Hard physical
            'force': 0.9, 'subdue': 0.9, 'physical': 0.8, 'struggle': 0.8,
            'resist': 0.9, 'fight': 0.9, 'tackle': 0.8
        },
        4: {  # Less lethal - Enhanced distinctive terms
            'taser': 1.0, 'spray': 1.0, 'baton': 1.0, 'less-lethal': 1.0,
            'projectile': 1.0, 'rubber': 1.0, 'bean bag': 1.0,
            'pepper spray': 1.0, 'OC spray': 1.0, 'chemical agent': 1.0,
            'impact munition': 1.0, 'stun gun': 1.0, 'tear gas': 1.0,
            'riot control': 1.0, 'crowd control': 0.9, 'incapacitate': 0.9,
            'non-lethal': 1.0, 'electroshock': 1.0
        },
        5: {  # Lethal - Enhanced distinctive terms
            'lethal': 1.0, 'firearm': 1.0, 'gun': 1.0, 'shoot': 1.0,
            'weapon': 1.0, 'deadly': 1.0, 'life-threatening': 1.0,
            'pistol': 1.0, 'rifle': 1.0, 'shotgun': 1.0, 'armed': 1.0,
            'discharge': 1.0, 'bullet': 1.0, 'ammunition': 1.0,
            'kill': 1.0, 'fatal': 1.0, 'death': 1.0, 'handgun': 1.0,
            'revolver': 1.0, 'magazine': 1.0, 'trigger': 1.0
        }
    }
    return force_terms.get(force_level, {})

def enhance_force_terms(text: str, force_terms: Dict[str, float]) -> str:
    """Enhance text with force-specific terms while maintaining natural language."""
    words = text.split()
    
    # Don't modify very short texts
    if len(words) < 3:
        return text
    
    # Add force-specific terms if they're not already present
    for term, importance in force_terms.items():
        if term not in text.lower() and random.random() < importance:
            # Choose insertion position - prefer later in the sentence for more natural flow
            insert_pos = random.randint(len(words) // 2, len(words))
            words.insert(insert_pos, term)
    
    # Add connecting words for better flow
    connectors = ['and', 'as', 'while', 'then']
    for i in range(len(words) - 1, 0, -1):
        if random.random() < 0.2:  # 20% chance to add connector
            words.insert(i, random.choice(connectors))
    
    return ' '.join(words)

def apply_law_enforcement_rules(text: str) -> str:
    """Apply domain-specific rules for law enforcement context."""
    # Standard procedure terminology
    procedure_terms = {
        'stop': 'cease movement',
        'get down': 'get on the ground',
        'hands up': 'raise your hands',
        'dont move': 'remain stationary',
        'come here': 'approach my position',
        'go away': 'leave the area',
        'run': 'flee',
        'fight': 'engage in physical confrontation'
    }
    
    # Replace casual terms with professional law enforcement terminology
    processed_text = text.lower()
    for casual, professional in procedure_terms.items():
        if casual in processed_text and random.random() < 0.7:  # 70% chance to replace
            processed_text = processed_text.replace(casual, professional)
    
    # Add standard police commands if not present
    commands = [
        'stop resisting',
        'show me your hands',
        'remain calm',
        'follow my instructions',
        'for your safety and mine'
    ]
    
    # Add a command if the text is command-like but doesn't have standard phrases
    if ('!' in text or 'stop' in text.lower()) and not any(cmd in text.lower() for cmd in commands):
        selected_command = random.choice(commands)
        processed_text = f"{processed_text}, {selected_command}"
    
    return processed_text

def advanced_augmentation(text: str, force_level: int) -> List[Tuple[str, int]]:
    augmented_samples = []
    
    # Base preprocessing
    text = text.strip().lower()
    
    # 1. Original sample
    augmented_samples.append((text, force_level))
    
    # 2. Force-specific augmentations
    force_terms = get_force_terms(force_level)
    augmented = enhance_force_terms(text, force_terms)
    if augmented != text:
        augmented_samples.append((augmented, force_level))
    
    # 3. Add law enforcement context
    augmented = apply_law_enforcement_rules(text)
    if augmented != text:
        augmented_samples.append((augmented, force_level))
    
    # 4. Contrastive examples (slightly modified versions)
    doc = nlp(text)
    
    # Synonym replacement
    for token in doc:
        if token.pos_ in ['VERB', 'ADJ']:
            synonyms = []
            for syn in wordnet.synsets(token.text):
                for lemma in syn.lemmas():
                    if lemma.name() != token.text:
                        synonyms.append(lemma.name())
            if synonyms:
                new_text = text.replace(token.text, random.choice(synonyms))
                augmented_samples.append((new_text, force_level))
    
    # 5. Word order variation (preserve meaning)
    if len(doc) > 3:
        chunks = list(doc.noun_chunks)
        if len(chunks) > 1:
            chunk_texts = [chunk.text for chunk in chunks]
            random.shuffle(chunk_texts)
            augmented_samples.append((' '.join(chunk_texts), force_level))
    
    return augmented_samples

class AdvancedTextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.force_terms = {
            'weapon': 5, 'gun': 5, 'knife': 5, 'firearm': 5,
            'taser': 4, 'spray': 4, 'baton': 4,
            'tackle': 3, 'strike': 3, 'hit': 3,
            'grab': 2, 'hold': 2, 'restrain': 2,
            'command': 1, 'order': 1, 'direct': 1,
            'talk': 0, 'verbal': 0, 'discuss': 0
        }
        self.resistance_terms = {'resist', 'fight', 'flee', 'run', 'escape', 'struggle', 'refuse'}
        self.compliance_terms = {'comply', 'cooperate', 'surrender', 'submit', 'follow', 'obey'}
    
    def preprocess(self, text: str) -> Dict[str, Union[str, List[str], bool, int]]:
        if not isinstance(text, str):
            return {
                'text': '',
                'force_indicators': [],
                'has_weapon': False,
                'has_resistance': False,
                'has_compliance': False,
                'force_level_hint': 0
            }
        
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract force indicators
        force_indicators = []
        max_force_level = 0
        
        for token in doc:
            if token.text in self.force_terms:
                force_indicators.append(token.text)
                max_force_level = max(max_force_level, self.force_terms[token.text])
        
        # Check for weapons, resistance, and compliance
        text_set = set(text.split())
        has_weapon = any(term in text for term in ['weapon', 'gun', 'knife', 'firearm', 'taser', 'baton'])
        has_resistance = bool(text_set.intersection(self.resistance_terms))
        has_compliance = bool(text_set.intersection(self.compliance_terms))
        
        # Adjust force level hint based on combinations
        force_level_hint = max_force_level
        if has_weapon:
            force_level_hint = max(force_level_hint, 4)
        if has_resistance and not has_compliance:
            force_level_hint = max(force_level_hint, 2)
        
        return {
            'text': text,
            'force_indicators': force_indicators,
            'has_weapon': has_weapon,
            'has_resistance': has_resistance,
            'has_compliance': has_compliance,
            'force_level_hint': force_level_hint
        }

class ForceDataCollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Initialize batch dictionary with empty lists
        batch_dict = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'force_indicators': [],
            'has_weapon': [],
            'has_resistance': [],
            'has_compliance': [],
            'force_level_hint': [],
            'labels': []
        }
        
        # Process each feature dictionary
        for f in features:
            # Each f is now a dictionary from ForceDataset.__getitem__
            batch_dict['input_ids'].append(f['input_ids'])
            batch_dict['attention_mask'].append(f['attention_mask'])
            batch_dict['token_type_ids'].append(f['token_type_ids'])
            batch_dict['force_indicators'].append(f['force_indicators'])
            batch_dict['has_weapon'].append(f['has_weapon'])
            batch_dict['has_resistance'].append(f['has_resistance'])
            batch_dict['has_compliance'].append(f['has_compliance'])
            batch_dict['force_level_hint'].append(f['force_level_hint'])
            batch_dict['labels'].append(f['labels'])
        
        # Convert lists to tensors
        for k in batch_dict:
            if len(batch_dict[k]) > 0:  # Check if the list is not empty
                batch_dict[k] = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in batch_dict[k]])
        
        return batch_dict

class ForceDataset(Dataset):
    def __init__(self, encodings, force_indicators, has_weapon, has_resistance, 
                 has_compliance, force_level_hints, labels):
        self.encodings = encodings
        self.force_indicators = force_indicators
        self.has_weapon = has_weapon
        self.has_resistance = has_resistance
        self.has_compliance = has_compliance
        self.force_level_hints = force_level_hints
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'force_indicators': self.force_indicators[idx],
            'has_weapon': self.has_weapon[idx],
            'has_resistance': self.has_resistance[idx],
            'has_compliance': self.has_compliance[idx],
            'force_level_hint': self.force_level_hints[idx],
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

def clean_and_balance_dataset(df):
    print("Starting dataset optimization...")
    
    # 1. Clean and standardize text
    print("Cleaning and standardizing text...")
    df['speech_turn'] = df['speech_turn'].astype(str)
    df['speech_turn'] = df['speech_turn'].apply(lambda x: x.strip().lower())
    
    # 2. Remove empty or invalid entries
    print("Removing invalid entries...")
    df = df.dropna(subset=['speech_turn', 'use_of_force_level'])
    df = df[df['speech_turn'].str.len() > 5]  # Remove very short texts
    
    # 3. Remove duplicates with smarter comparison
    print("Removing duplicates...")
    # Normalize text for comparison
    df['normalized_text'] = df['speech_turn'].apply(lambda x: ' '.join(sorted(x.split())))
    df = df.drop_duplicates(subset=['normalized_text', 'use_of_force_level'])
    df = df.drop(columns=['normalized_text'])
    
    # 4. Remove level 0 and keep only valid force levels (1-5)
    print("Removing level 0 and standardizing force levels...")
    df = df[df['use_of_force_level'].between(1, 5)]  # Keep only levels 1-5
    
    # 5. Print final dataset statistics
    print("\nDataset optimization complete!")
    print("\nFinal class distribution:")
    print(df['use_of_force_level'].value_counts().sort_index())
    print(f"\nTotal samples: {len(df)}")
    
    # 6. Shuffle the final dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def prepare_data(logger):
    logger.info("Loading dataset...")
    df = pd.read_csv('Data.csv')
    
    # Print initial class distribution
    logger.info("\nInitial class distribution:")
    logger.info(df['use_of_force_level'].value_counts().sort_index())
    
    # Clean dataset but don't balance
    df = clean_and_balance_dataset(df)
    logger.info(f"\nDataset size after cleaning: {len(df)}")
    
    # Print final class distribution
    logger.info("\nFinal class distribution:")
    logger.info(df['use_of_force_level'].value_counts().sort_index())
    
    # Print sample texts for each class
    logger.info("\nSample texts for each force level:")
    for level in range(1, 6):  # Changed range to 1-5
        samples = df[df['use_of_force_level'] == level]['speech_turn'].head(2).tolist()
        logger.info(f"\nForce Level {level} samples:")
        for sample in samples:
            logger.info(f"- {sample}")
    
    # Initialize tokenizer and text preprocessor
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    text_preprocessor = AdvancedTextPreprocessor()
    
    # Prepare features
    texts = df['speech_turn'].tolist()
    labels = df['use_of_force_level'].tolist()
    
    # Generate additional features from text content
    force_indicators = []
    has_weapon = []
    has_resistance = []
    has_compliance = []
    force_level_hints = []
    
    logger.info("Generating additional features from text...")
    for text in texts:
        features = text_preprocessor.preprocess(text)
        force_indicators.append(len(features['force_indicators']))
        has_weapon.append(1 if features['has_weapon'] else 0)
        has_resistance.append(1 if features['has_resistance'] else 0)
        has_compliance.append(1 if features['has_compliance'] else 0)
        force_level_hints.append(features['force_level_hint'])
    
    # Tokenize all texts
    logger.info("Tokenizing texts...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Convert to tensors
    labels = torch.tensor(labels)
    force_indicators = torch.tensor(force_indicators)
    has_weapon = torch.tensor(has_weapon)
    has_resistance = torch.tensor(has_resistance)
    has_compliance = torch.tensor(has_compliance)
    force_level_hints = torch.tensor(force_level_hints)
    
    # Create dataset
    dataset = ForceDataset(
        encodings=encodings,
        force_indicators=force_indicators,
        has_weapon=has_weapon,
        has_resistance=has_resistance,
        has_compliance=has_compliance,
        force_level_hints=force_level_hints,
        labels=labels
    )
    
    # Split dataset with a larger portion for training
    train_size = int(0.85 * len(dataset))  # Using 85% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    return train_dataset, val_dataset

def create_efficient_dataloader(dataset, batch_size, num_workers=None):
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count() - 1, 8)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

def augment_text(text: str, force_level: int) -> Tuple[List[str], List[int]]:
    """
    Augment text with force-specific variations and return list of augmented texts and their force levels.
    """
    augmented_texts = [text]  # Start with original text
    force_levels = [force_level]  # Start with original force level
    
    # Get force-specific terms
    force_terms = get_force_terms(force_level)
    
    # 1. Add force-specific terms
    enhanced = enhance_force_terms(text, force_terms)
    if enhanced != text:
        augmented_texts.append(enhanced)
        force_levels.append(force_level)
    
    # 2. Apply law enforcement rules
    professional = apply_law_enforcement_rules(text)
    if professional != text:
        augmented_texts.append(professional)
        force_levels.append(force_level)
    
    # 3. Synonym replacement (up to 2 variations)
    doc = nlp(text)
    for token in doc:
        if token.pos_ in ['VERB', 'ADJ']:
            synonyms = []
            for syn in wordnet.synsets(token.text):
                for lemma in syn.lemmas():
                    if lemma.name() != token.text:
                        synonyms.append(lemma.name())
            if synonyms and len(augmented_texts) < 5:  # Limit variations
                new_text = text.replace(token.text, random.choice(synonyms))
                augmented_texts.append(new_text)
                force_levels.append(force_level)
    
    # 4. Word order variation (if applicable)
    if len(doc) > 3:
        chunks = list(doc.noun_chunks)
        if len(chunks) > 1:
            chunk_texts = [chunk.text for chunk in chunks]
            random.shuffle(chunk_texts)
            reordered = ' '.join(chunk_texts)
            if reordered != text:
                augmented_texts.append(reordered)
                force_levels.append(force_level)
    
    return augmented_texts, force_levels

def main():
    # Set up logging with more detailed configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', mode='w'),  # 'w' mode to start fresh
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Also capture warnings in the log
    logging.captureWarnings(True)

    try:
        logger.info("Starting training process...")
        logger.info("Loading and preprocessing data...")
        
        # Prepare data
        train_dataset, val_dataset = prepare_data(logger)
        logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
        
        # Initialize model with 5 classes
        logger.info("Initializing model...")
        config = ForceClassificationConfig(
            num_labels=5,  # Changed from 6 to 5 classes
            hidden_size=768,
            num_attention_heads=12,
            feature_fusion_size=256,
            dropout_rate=0.2
        )
        model = ImprovedForceClassificationEnsemble(config)
        logger.info("Model initialized successfully")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create data collator
        data_collator = ForceDataCollator(tokenizer)
        
        # Set up training arguments with more detailed logging
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=15,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_ratio=0.1,
            weight_decay=0.02,
            logging_dir="./logs",
            logging_steps=10,  # More frequent logging
            save_steps=200,
            eval_steps=200,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            learning_rate=2e-5,
            gradient_accumulation_steps=2,
            fp16=False,
            dataloader_drop_last=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            group_by_length=True,
            save_total_limit=3,
            logging_first_step=True,  # Log the first training step
            report_to=["tensorboard"],  # Enable tensorboard logging
        )
        
        # Initialize curriculum scheduler
        curriculum_scheduler = CurriculumScheduler(train_dataset, num_epochs=training_args.num_train_epochs)
        
        # Set up trainer
        logger.info("Setting up trainer...")
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            curriculum_scheduler=curriculum_scheduler,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )
        logger.info("Trainer setup complete")
        
        # Train the model
        logger.info("Beginning training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model("./final_model")
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error("Full traceback:", exc_info=True)  # This will log the full traceback
        raise  # Re-raise the exception after logging

if __name__ == "__main__":
    main() 