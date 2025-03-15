# Phishing Email Detection AI

## Sample emails sourced from https://github.com/rf-peixoto/phishing_pot

Project is an artificial intelligence (AI) phishing email detection system. It contains two AI models:
1. **AI Email Generator (Llama-2)** - Generates phishing and normal emails.
2. **AI Email Detector (Hybrid CNN + Transformer)** - Determines if an email is phishing or not.
3. **Score System** - Tracks the performance of both AI models in a reinforcement learning environment.

## Features
- **Phishing Email Generation** with Llama-2.
- **Email Detection** with hybrid CNN + Transformer model.
- **Dataset Loading** from email files with the `.eml` extension.
- **Scoring System** for AI performance measurement.

## Project Structure
```
phishing_ai_project/
│── generator.py        # AI generates phishing emails (Llama-2)
│── detector.py         # AI detects phishing emails (Hybrid CNN + Transformer)
│── score_system.py     # Reinforcement learning-based scoring system
│── phishing_sim.py     # Main script to run the simulation
│── email/              # Folder containing sample .eml email dataset
│── requirements.txt    # Dependencies list
```

## Installation
### 1️⃣ Clone the Repository
```
git clone https://github.com/your-repo/phishing-ai.git
cd phishing-ai
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

## Running the Project
### 3️⃣ Run the Simulation
```
python phishing_sim.py
```
- The script will load emails from the `email/` folder.
- It will randomly generate emails and evaluate phishing detection accuracy.
- The score system will update based on AI performance.

## Expected Output
```
Loading email dataset...
Total emails loaded: 1000
Generating phishing email...
Email Generated: "Dear user, please update your password now!"
AI Detector Prediction: Phishing
Scores: Detector=1, Generator=0
```

## Customizing the Project
- Add more `.eml` files in the `email/` folder to improve dataset quality.
- Train the **detector model** on real phishing emails.
- Modify `generator.py` to fine-tune Llama-2 email generation behavior.

## License
MIT License


