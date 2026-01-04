"""
Email Spam Detection AI System
Unsupervised learning approach using Autoencoder
"""


import os
from dataloader import create_dataloader, EmailDataLoader
from model import create_model, EmailAutoencoder
from train import train_autoencoder, load_checkpoint, plot_reconstruction_examples
from predict import EmailPredictor, predict_email

__version__ = '1.0.0'
__author__ = 'AI Spam Detection System'

class SpamDetectionAI:
    """Main AI system interface for email spam detection"""
    
    def __init__(self, data_folder='data/email', model_folder='data'):
        """
        Initialize AI system
        
        Args:
            data_folder: Folder containing .eml files
            model_folder: Folder to save/load models and analysis
        """
        self.data_folder = data_folder
        self.model_folder = model_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dataloader = None
        self.model = None
        self.predictor = None
        self.trained = False
        
        print("="*60)
        print("Email Spam Detection AI System v{}".format(__version__))
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Data folder: {self.data_folder}")
        print(f"Model folder: {self.model_folder}")
        print("="*60 + "\n")
    
    def prepare_data(self, batch_size=32, min_freq=2):
        """
        Load and prepare email data
        
        Args:
            batch_size: Batch size for training
            min_freq: Minimum word frequency for vocabulary
        """
        print("[AI] Preparing data...")
        
        self.dataloader_obj, self.loader, self.email_data = create_dataloader(
            data_folder=self.data_folder,
            batch_size=batch_size,
            min_freq=min_freq
        )
        
        if self.dataloader_obj is None:
            raise Exception("Failed to load data. Check data folder.")
        
        print(f"[AI] Data preparation complete")
        print(f"[AI] Vocabulary size: {self.loader.vocab_size}")
        print(f"[AI] Total emails: {len(self.email_data)}\n")
        
        return self.dataloader_obj
    
    def build_model(self, embedding_dim=128, hidden_dim=256, latent_dim=64):
        """
        Build autoencoder model
        
        Args:
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            latent_dim: Latent space dimension
        """
        print("[AI] Building model...")
        
        if self.loader is None:
            raise Exception("Data not prepared. Call prepare_data() first.")
        
        self.model = create_model(
            vocab_size=self.loader.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        print(f"[AI] Model built successfully\n")
        
        return self.model
    
    def train(self, epochs=50, lr=0.001):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
        """
        print("[AI] Starting training...")
        
        if self.model is None or self.dataloader_obj is None:
            raise Exception("Model or data not ready. Call prepare_data() and build_model() first.")
        
        train_losses, recon_losses, anomaly_losses = train_autoencoder(
            model=self.model,
            dataloader=self.dataloader_obj,
            epochs=epochs,
            lr=lr,
            device=self.device,
            save_dir=self.model_folder
        )
        
        # Plot reconstruction examples
        plot_reconstruction_examples(
            self.model, 
            self.dataloader_obj, 
            num_examples=5, 
            save_dir=self.model_folder
        )
        
        self.trained = True
        
        print(f"\n[AI] Training complete!")
        print(f"[AI] Final loss: {train_losses[-1]:.4f}")
        print(f"[AI] Model saved to {self.model_folder}/model_final.pth")
        print(f"[AI] Analysis saved to {self.model_folder}/analysis/\n")
        
        return train_losses, recon_losses, anomaly_losses
    
    def load_trained_model(self, model_path=None):
        """
        Load a trained model
        
        Args:
            model_path: Path to model file (default: data/model_final.pth)
        """
        if model_path is None:
            model_path = os.path.join(self.model_folder, 'model_final.pth')
        
        print(f"[AI] Loading trained model from {model_path}...")
        
        # Load dataloader first
        dataloader_path = os.path.join(self.model_folder, 'dataloader.pkl')
        self.loader = EmailDataLoader()
        self.loader.load(dataloader_path)
        
        # Build model structure
        self.model = EmailAutoencoder(
            vocab_size=self.loader.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            latent_dim=64
        )
        
        # Load weights
        self.model = load_checkpoint(self.model, model_path, self.device)
        self.trained = True
        
        print(f"[AI] Model loaded successfully\n")
        
        return self.model
    
    def predict(self, email_path, set_threshold=True):
        """
        Predict if an email is spam
        
        Args:
            email_path: Path to .eml file
            set_threshold: Whether to calculate threshold from training data
        """
        print(f"[AI] Predicting email: {email_path}")
        
        if not self.trained:
            raise Exception("Model not trained. Call train() or load_trained_model() first.")
        
        # Initialize predictor
        if self.predictor is None:
            model_path = os.path.join(self.model_folder, 'model_final.pth')
            dataloader_path = os.path.join(self.model_folder, 'dataloader.pkl')
            
            self.predictor = EmailPredictor(model_path, dataloader_path, self.device)
            
            if set_threshold:
                self.predictor.set_threshold(self.data_folder)
        
        # Predict
        result = self.predictor.predict_file(email_path, return_details=True)
        
        if result:
            self.predictor.print_prediction(result)
        
        return result
    
    def predict_batch(self, email_folder=None):
        """
        Predict all emails in a folder
        
        Args:
            email_folder: Folder containing .eml files (default: data_folder)
        """
        if email_folder is None:
            email_folder = self.data_folder
        
        print(f"[AI] Batch prediction on folder: {email_folder}")
        
        if not self.trained:
            raise Exception("Model not trained. Call train() or load_trained_model() first.")
        
        if self.predictor is None:
            model_path = os.path.join(self.model_folder, 'model_final.pth')
            dataloader_path = os.path.join(self.model_folder, 'dataloader.pkl')
            
            self.predictor = EmailPredictor(model_path, dataloader_path, self.device)
            self.predictor.set_threshold(self.data_folder)
        
        results = self.predictor.predict_batch(email_folder)
        
        # Print summary
        spam_count = sum(1 for r in results if r['is_spam'])
        normal_count = len(results) - spam_count
        
        print(f"\n[AI] Batch prediction complete:")
        print(f"[AI] Total: {len(results)}, Spam: {spam_count}, Normal: {normal_count}\n")
        
        return results
    
    def run_full_pipeline(self, epochs=50, test_email=None):
        """
        Run complete pipeline: prepare data, build model, train, and predict
        
        Args:
            epochs: Number of training epochs
            test_email: Email file to test prediction (optional)
        """
        print("[AI] Running full pipeline...\n")
        
        # 1. Prepare data
        self.prepare_data()
        
        # 2. Build model
        self.build_model()
        
        # 3. Train
        self.train(epochs=epochs)
        
        # 4. Predict (if test email provided)
        if test_email and os.path.exists(test_email):
            print("[AI] Testing prediction on sample email...")
            self.predict(test_email)
        
        print("[AI] Full pipeline complete!\n")


# Convenience functions for quick usage
def train_new_model(data_folder='data/email', epochs=50, batch_size=32):
    """Quick function to train a new model"""
    ai = SpamDetectionAI(data_folder=data_folder)
    ai.prepare_data(batch_size=batch_size)
    ai.build_model()
    ai.train(epochs=epochs)
    return ai


def predict_with_trained_model(email_path, model_folder='data'):
    """Quick function to predict with trained model"""
    ai = SpamDetectionAI(model_folder=model_folder)
    ai.load_trained_model()
    return ai.predict(email_path)


# Example usage
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Email Spam Detection AI System v{}              ║
╚══════════════════════════════════════════════════════════════╝

Example Usage:
--------------

1. Full pipeline (train and predict):
   >>> from __init__ import SpamDetectionAI
   >>> ai = SpamDetectionAI(data_folder='data/email')
   >>> ai.run_full_pipeline(epochs=50, test_email='sample.eml')

2. Train new model:
   >>> ai = SpamDetectionAI()
   >>> ai.prepare_data(batch_size=32)
   >>> ai.build_model()
   >>> ai.train(epochs=50)

3. Load and predict:
   >>> ai = SpamDetectionAI()
   >>> ai.load_trained_model()
   >>> result = ai.predict('test_email.eml')

4. Batch prediction:
   >>> ai = SpamDetectionAI()
   >>> ai.load_trained_model()
   >>> results = ai.predict_batch('test_folder')

For AI vs AI battle, use this interface to call predictions!
    """.format(__version__))