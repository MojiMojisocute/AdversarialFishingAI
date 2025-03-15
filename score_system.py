class ScoreSystem:
    def __init__(self):
        self.detector_score = 0
        self.generator_score = 0

    def update_scores(self, is_phishing, detected_as_phishing):
        if is_phishing and detected_as_phishing:
            self.detector_score += 1
        elif not is_phishing and not detected_as_phishing:
            self.detector_score += 1
        elif not is_phishing and detected_as_phishing:
            self.detector_score -= 1
            self.generator_score += 1
        else:
            self.generator_score += 1

    def get_scores(self):
        return self.detector_score, self.generator_score
