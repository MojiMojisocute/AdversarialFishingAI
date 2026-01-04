import torch
import numpy as np
import os
from dataloader import EmailDataLoader
from model import EmailAutoencoder

class EmailPredictor:
    """Predict if email is spam using trained autoencoder"""
    
    def __init__(self, model_path='data/model_final.pth', 
                 dataloader_path='data/dataloader.pkl',
                 device='cuda'):
          
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"[Predictor] Using device: {self.device}")
        
        self.dataloader = EmailDataLoader()
        self.dataloader.load(dataloader_path)
        
        self.model = EmailAutoencoder(
            vocab_size=self.dataloader.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            latent_dim=64
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[Predictor] Model loaded from {model_path}")
        
        self.threshold = None
        
    def set_threshold(self, normal_emails_folder='data/email', percentile=95):
        """Calculate threshold from normal emails"""
        print(f"[Predictor] Calculating threshold from {normal_emails_folder}")
        
        errors = []
        
        eml_files = [f for f in os.listdir(normal_emails_folder) if f.endswith('.eml')]
        
        for filename in eml_files[:100]: 
            file_path = os.path.join(normal_emails_folder, filename)
            
            try:
                headers, body = self.dataloader.read_eml_file(file_path)
                if headers and body:
                    text = f"{headers['subject']} {headers['from']} {body}"
                    cleaned = self.dataloader.clean_text(text)
                    sequence = self.dataloader.text_to_sequence(cleaned)
                    
                    with torch.no_grad():
                        seq_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                        error = self.model.get_reconstruction_error(seq_tensor)
                        errors.append(error.item())
            except:
                continue
        
        if errors:
            self.threshold = np.percentile(errors, percentile)
            print(f"[Predictor] Threshold set to {self.threshold:.4f} ({percentile}th percentile)")
        else:
            self.threshold = 5.0
            print(f"[Predictor] Using default threshold: {self.threshold}")
        
        return self.threshold
    
    def predict_file(self, email_path, return_details=False):
        """Predict if email file is spam"""
        
        headers, body = self.dataloader.read_eml_file(email_path)
        
        if not headers or not body:
            return None
        
        text = f"{headers['subject']} {headers['from']} {body}"
        cleaned_text = self.dataloader.clean_text(text)
        sequence = self.dataloader.text_to_sequence(cleaned_text)
        
        seq_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            reconstructed, latent, anomaly_score = self.model(seq_tensor)
            recon_error = self.model.get_reconstruction_error(seq_tensor).item()
        
        if self.threshold is None:
            is_spam = anomaly_score.item() > 0.5
            confidence = anomaly_score.item()
        else:
            is_spam = recon_error > self.threshold
            confidence = min(recon_error / (self.threshold * 2), 1.0) if is_spam else 1.0 - (recon_error / self.threshold)
        
        result = {
            'is_spam': is_spam,
            'confidence': confidence,
            'recon_error': recon_error,
            'anomaly_score': anomaly_score.item(),
            'label': 'SPAM' if is_spam else 'NORMAL'
        }
        
        if return_details:
            result['headers'] = headers
            result['text_preview'] = cleaned_text[:200]
            result['latent_vector'] = latent.cpu().numpy()
        
        return result
    
    def predict_batch(self, email_folder):
        """Predict all emails in a folder"""
        
        results = []
        
        eml_files = [f for f in os.listdir(email_folder) if f.endswith('.eml')]
        print(f"[Predictor] Predicting {len(eml_files)} emails...")
        
        for filename in eml_files:
            file_path = os.path.join(email_folder, filename)
            
            try:
                result = self.predict_file(file_path)
                if result:
                    result['filename'] = filename
                    results.append(result)
            except Exception as e:
                print(f"[Error] Failed to predict {filename}: {e}")
        
        return results
    
    def print_prediction(self, result):
        """Print prediction result in nice format"""
        
        print("\n" + "="*60)
        print("EMAIL PREDICTION RESULT")
        print("="*60)
        
        if result.get('headers'):
            print(f"\nFrom: {result['headers']['from']}")
            print(f"Subject: {result['headers']['subject']}")
            print(f"Date: {result['headers']['date']}")
        
        print(f"\n{'SPAM DETECTED' if result['is_spam'] else '✓ NORMAL EMAIL'}")
        print(f"\nConfidence: {result['confidence']*100:.2f}%")
        print(f"Reconstruction Error: {result['recon_error']:.4f}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        
        if self.threshold:
            print(f"Threshold: {self.threshold:.4f}")
        
        if result.get('text_preview'):
            print(f"\nText Preview:")
            print(result['text_preview'])
        
        print("="*60 + "\n")


def predict_email(email_path, model_path='data/model_final.pth', 
                  dataloader_path='data/dataloader.pkl'):
    """Quick prediction function"""
    
    predictor = EmailPredictor(model_path, dataloader_path)
    predictor.set_threshold()
    
    result = predictor.predict_file(email_path, return_details=True)
    
    if result:
        predictor.print_prediction(result)
        return result
    else:
        print("[Error] Could not predict email")
        return None


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Email Spam Predictor")
    print("="*60)
    
    if len(sys.argv) > 1:
        email_path = sys.argv[1]
    else:
        email_path = 'data/email/sample-1.eml'
    
    if not os.path.exists(email_path):
        print(f"[Error] File not found: {email_path}")
        print("\nUsage: python predict.py <email_file.eml>")
    else:
        result = predict_email(email_path)
        
        if result:
            print(f"\nFinal prediction: {result['label']}")