import os
import email
from email import policy
import random
import torch
from generator import generate_email
from detector import model
from score_system import ScoreSystem
from sklearn.feature_extraction.text import TfidfVectorizer

def load_emails_from_folder(folder_path):
    emails = []
    filenames = os.listdir(folder_path)
    
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            msg = email.message_from_file(f, policy=policy.default)
            emails.append(msg.get_body(preferencelist=('plain')).get_content())
    
    return emails

email_dataset = load_emails_from_folder("email/")
num_samples = len(email_dataset)

labels = [random.choice([0, 1]) for _ in range(num_samples)]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(email_dataset).toarray()

score_system = ScoreSystem()

def phishing_simulation():
    prompt = random.choice(["Generate a realistic phishing email.", "Generate a normal email."])
    generated_email = generate_email(prompt)
    is_phishing = "phishing" in prompt

    detected_as_phishing = model(torch.tensor(vectorizer.transform([generated_email]).toarray(), dtype=torch.float)).argmax().item() == 1
    score_system.update_scores(is_phishing, detected_as_phishing)
    print("Scores:", score_system.get_scores())

phishing_simulation()
