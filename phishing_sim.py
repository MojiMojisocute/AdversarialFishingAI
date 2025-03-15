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
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="latin-1", errors="ignore") as file:
            msg = email.message_from_file(file, policy=policy.default)

            if msg.is_multipart():
                body = ""
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

            emails.append(body)
    
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

    # ใช้ sparse matrix โดยไม่ต้องแปลงเป็น array
    email_vector = vectorizer.transform([generated_email])
    detected_as_phishing = model(torch.tensor(email_vector.toarray(), dtype=torch.float)).argmax().item() == 1

    score_system.update_scores(is_phishing, detected_as_phishing)
    print("Scores:", score_system.get_scores())

phishing_simulation()
