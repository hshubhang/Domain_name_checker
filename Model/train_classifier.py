import json
import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
import gc

class BusinessDescriptionClassifier:
    """
    Train a lightweight classifier to detect gibberish vs legitimate business descriptions.
    Uses the existing augmented dataset v2 with source field labels.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the classifier trainer.
        
        Args:
            model_name: Pretrained model to fine-tune for classification
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Initializing Business Description Classifier")
        print(f"📦 Base model: {model_name}")
        print(f"🖥️  Device: {self.device}")
        
    def load_and_prepare_data(self, dataset_path: str = "../data/augmented_dataset_v3.jsonl") -> Tuple[List[str], List[int]]:
        """
        Load data from the augmented dataset and create classification labels.
        
        Args:
            dataset_path: Path to the augmented dataset v2
            
        Returns:
            Tuple of (texts, labels) where labels are 1=legitimate, 0=gibberish
        """
        print(f"📥 Loading data from {dataset_path}")
        
        texts = []
        labels = []
        
        # Count by source for verification
        source_counts = {"original": 0, "paraphrase": 0, "gibberish": 0}
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        item = json.loads(line)
                        business_desc = item.get("business_description", "")
                        source = item.get("source", "")
                        
                        if not business_desc or not source:
                            print(f"Warning: Line {line_num} missing required fields")
                            continue
                        
                        texts.append(business_desc)
                        
                        # Map source to binary labels
                        if source in ["original", "paraphrase"]:
                            labels.append(1)  # Legitimate
                            source_counts[source] += 1
                        elif source == "gibberish":
                            labels.append(0)  # Gibberish
                            source_counts["gibberish"] += 1
                        else:
                            print(f"Warning: Unknown source '{source}' on line {line_num}")
                            continue
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"❌ Error: Could not find dataset file {dataset_path}")
            return [], []
        
        print(f"✅ Loaded {len(texts)} examples:")
        print(f"   📝 Original: {source_counts['original']}")
        print(f"   🔄 Paraphrase: {source_counts['paraphrase']}")
        print(f"   🚫 Gibberish: {source_counts['gibberish']}")
        print(f"   ➡️  Legitimate (1): {sum(labels)}")
        print(f"   ➡️  Gibberish (0): {len(labels) - sum(labels)}")
        
        return texts, labels
    
    def balance_dataset(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """
        Balance the dataset using targeted undersampling that preserves newly added generic businesses.
        
        Args:
            texts: List of business descriptions
            labels: List of binary labels (1=legitimate, 0=gibberish)
            
        Returns:
            Balanced texts and labels
        """
        print(f"🔄 Balancing dataset with targeted undersampling...")
        
        # Separate classes
        legitimate_indices = [i for i, label in enumerate(labels) if label == 1]
        gibberish_indices = [i for i, label in enumerate(labels) if label == 0]
        
        legitimate_texts = [texts[i] for i in legitimate_indices]
        gibberish_texts = [texts[i] for i in gibberish_indices]
        
        print(f"   Before balancing:")
        print(f"   📝 Legitimate: {len(legitimate_texts)} examples")
        print(f"   🚫 Gibberish: {len(gibberish_texts)} examples")
        
        # Target size for balanced dataset
        target_size = min(len(legitimate_texts), len(gibberish_texts))
        print(f"   🎯 Target size per class: {target_size}")
        
        # For legitimate examples: preserve the LAST 50 (likely our new generic businesses)
        # and randomly sample from the rest
        if len(legitimate_texts) > target_size:
            # Assume the last 50 legitimate examples are our new generic businesses
            num_generic_preserve = min(50, len(legitimate_texts))
            preserved_generic = legitimate_texts[-num_generic_preserve:]  # Last 50
            remaining_legitimate = legitimate_texts[:-num_generic_preserve]  # All except last 50
            
            # How many more do we need from the remaining pool?
            additional_needed = target_size - num_generic_preserve
            
            if additional_needed > 0 and len(remaining_legitimate) > 0:
                # Randomly sample from the remaining legitimate examples
                np.random.seed(42)
                if additional_needed >= len(remaining_legitimate):
                    sampled_remaining = remaining_legitimate
                else:
                    indices = np.random.choice(len(remaining_legitimate), additional_needed, replace=False)
                    sampled_remaining = [remaining_legitimate[i] for i in indices]
                
                final_legitimate = preserved_generic + sampled_remaining
                print(f"   ✅ Preserved {len(preserved_generic)} new generic businesses")
                print(f"   ✅ Sampled {len(sampled_remaining)} from existing legitimate examples")
            else:
                final_legitimate = preserved_generic[:target_size]
                print(f"   ✅ Using only {len(final_legitimate)} preserved generic businesses")
        else:
            final_legitimate = legitimate_texts
            print(f"   ✅ Keeping all {len(final_legitimate)} legitimate examples")
        
        # For gibberish: random undersample if needed
        if len(gibberish_texts) > target_size:
            np.random.seed(42)
            indices = np.random.choice(len(gibberish_texts), target_size, replace=False)
            final_gibberish = [gibberish_texts[i] for i in indices]
        else:
            final_gibberish = gibberish_texts
        
        # Combine and shuffle
        balanced_texts = final_legitimate + final_gibberish
        balanced_labels = [1] * len(final_legitimate) + [0] * len(final_gibberish)
        
        # Shuffle together
        combined = list(zip(balanced_texts, balanced_labels))
        np.random.shuffle(combined)
        balanced_texts, balanced_labels = zip(*combined)
        
        print(f"   After targeted balancing:")
        print(f"   📝 Legitimate: {sum(balanced_labels)} examples")
        print(f"   🚫 Gibberish: {len(balanced_labels) - sum(balanced_labels)} examples")
        print(f"   📊 Total: {len(balanced_labels)} examples")
        print(f"   🎯 Balance ratio: {sum(balanced_labels) / len(balanced_labels) * 100:.1f}% legitimate")
        
        return list(balanced_texts), list(balanced_labels)
    
    def prepare_dataset(self, texts: List[str], labels: List[int], test_size: float = 0.2, balance_data: bool = True) -> Tuple[Dataset, Dataset]:
        """
        Prepare train/test datasets for training.
        
        Args:
            texts: List of business descriptions
            labels: List of binary labels (1=legitimate, 0=gibberish)
            test_size: Fraction of data to use for testing
            balance_data: Whether to balance the dataset by undersampling majority class
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        print(f"🔄 Preparing datasets (test_size={test_size}, balance_data={balance_data})")
        
        # Balance dataset if requested
        if balance_data:
            texts, labels = self.balance_dataset(texts, labels)
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"📊 Data split:")
        print(f"   Training: {len(train_texts)} examples")
        print(f"   Testing: {len(test_texts)} examples")
        
        # Load tokenizer
        print(f"📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize data
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=128  # Shorter than domain generation task
            )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            "text": train_texts,
            "labels": train_labels
        })
        
        test_dataset = Dataset.from_dict({
            "text": test_texts,
            "labels": test_labels
        })
        
        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        return train_dataset, test_dataset
    
    def train_classifier(self, train_dataset: Dataset, test_dataset: Dataset, output_dir: str = "./classifier_model_v3/"):
        """
        Train the classification model with class balancing and anti-overfitting techniques.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset  
            output_dir: Directory to save the trained model
        """
        print(f"🚀 Starting classifier training...")
        
        # Calculate class weights for balancing
        train_labels = train_dataset['labels']
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"📊 Class weights: {class_weights_dict}")
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,  # Binary classification
            id2label={0: "gibberish", 1: "legitimate"},
            label2id={"gibberish": 0, "legitimate": 1}
        )
        
        # Custom trainer class to handle class weights
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                # Apply class weights
                weight_tensor = torch.tensor([class_weights_dict[i] for i in labels.cpu().numpy()], 
                                           dtype=torch.float, device=labels.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(list(class_weights_dict.values()), 
                                                                       dtype=torch.float, device=labels.device))
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        # Training arguments with anti-overfitting measures
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=1e-5,  # Lower learning rate to prevent overfitting
            per_device_train_batch_size=8,  # Smaller batch size
            per_device_eval_batch_size=8,
            num_train_epochs=10,  # More epochs but with early stopping
            weight_decay=0.1,  # Stronger regularization
            eval_strategy="steps",  # Evaluate more frequently
            eval_steps=50,  # Evaluate every 50 steps
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",  # Use F1 score instead of accuracy
            greater_is_better=True,
            logging_steps=25,
            warmup_steps=50,
            report_to=None,  # Disable wandb/tensorboard
            save_total_limit=3,
            dataloader_drop_last=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Enhanced metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            f1_macro = f1_score(labels, predictions, average='macro')
            f1_weighted = f1_score(labels, predictions, average='weighted')
            precision_macro = precision_score(labels, predictions, average='macro')
            recall_macro = recall_score(labels, predictions, average='macro')
            
            # Per-class metrics
            f1_per_class = f1_score(labels, predictions, average=None)
            precision_per_class = precision_score(labels, predictions, average=None)
            recall_per_class = recall_score(labels, predictions, average=None)
            
            return {
                "accuracy": accuracy,
                "f1": f1_macro,
                "f1_weighted": f1_weighted,
                "precision": precision_macro,
                "recall": recall_macro,
                "f1_gibberish": f1_per_class[0],
                "f1_legitimate": f1_per_class[1],
                "precision_gibberish": precision_per_class[0],
                "precision_legitimate": precision_per_class[1],
                "recall_gibberish": recall_per_class[0],
                "recall_legitimate": recall_per_class[1],
            }
        
        # Initialize trainer with early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,  # Stop if no improvement for 3 evaluations
            early_stopping_threshold=0.001  # Minimum improvement threshold
        )
        
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
        )
        
        # Train
        print(f"⏰ Training starting...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Training completed! Model saved to {output_dir}")
        
        return trainer
    
    def evaluate_model(self, trainer, test_dataset: Dataset):
        """
        Evaluate the trained model and print detailed metrics.
        
        Args:
            trainer: Trained model trainer
            test_dataset: Test dataset for evaluation
        """
        print(f"📊 Evaluating model performance...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        
        print(f"\n🎯 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(
            true_labels, 
            pred_labels, 
            target_names=["gibberish", "legitimate"],
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        print(f"\n🔍 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"              Gib    Legit")
        print(f"   Actual Gib  {cm[0][0]:3d}     {cm[0][1]:3d}")
        print(f"         Legit {cm[1][0]:3d}     {cm[1][1]:3d}")
        
        # Calculate precision, recall for gibberish detection
        gibberish_precision = cm[0][0] / (cm[0][0] + cm[1][0]) if (cm[0][0] + cm[1][0]) > 0 else 0
        gibberish_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        
        print(f"\n🚫 Gibberish Detection Metrics:")
        print(f"   Precision: {gibberish_precision:.4f} (% of gibberish predictions that were correct)")
        print(f"   Recall: {gibberish_recall:.4f} (% of actual gibberish that was detected)")
        
        return accuracy
    
    def test_classifier(self, model_dir: str = "./classifier_model_v3/"):
        """
        Test the trained classifier with some examples.
        
        Args:
            model_dir: Directory containing the trained model
        """
        print(f"🧪 Testing classifier with examples...")
        
        # Load trained model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(self.device)
        
        # Test examples
        test_examples = [
            "A sustainable fashion platform connecting conscious consumers with ethical brands",
            "xkjfhe983 fmxlk234 postgres ERROR: connection timeout 127.0.0.1:5432",
            "An AI-powered fintech startup providing blockchain solutions",
            "SELECT * FROM users WHERE password = 'admin123';",
            "A traditional Italian restaurant serving authentic pasta dishes",
            "Random keyboard mash: asdfghjkl qwertyuiop zxcvbnm",
            "A mobile app for booking fitness classes in your neighborhood"
        ]
        
        print(f"\n🔍 Test Results:")
        
        for text in test_examples:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            label = "legitimate" if predicted_class == 1 else "gibberish"
            print(f"   {confidence:.3f} | {label:10} | {text[:50]}...")
        
        return model, tokenizer


def main():
    """Main function to train the business description classifier."""
    
    # Initialize classifier
    classifier = BusinessDescriptionClassifier()
    
    # Load and prepare data
    texts, labels = classifier.load_and_prepare_data()
    
    if not texts:
        print("❌ No data loaded. Exiting.")
        return
    
    # Prepare datasets
    train_dataset, test_dataset = classifier.prepare_dataset(texts, labels)
    
    # Train classifier
    trainer = classifier.train_classifier(train_dataset, test_dataset)
    
    # Evaluate model
    accuracy = classifier.evaluate_model(trainer, test_dataset)
    
    # Test with examples
    classifier.test_classifier()
    
    print(f"\n🎉 Classifier training completed!")
    print(f"📊 Final accuracy: {accuracy:.4f}")
    print(f"💾 Model saved to: ./classifier_model_v3/")
    print(f"🔄 Next step: Create model_v4.py for sequential pipeline")


if __name__ == "__main__":
    main() 