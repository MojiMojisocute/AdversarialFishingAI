import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class HybridClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(HybridClassifier, self).__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, num_classes)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedded = bert_output.last_hidden_state.permute(0, 2, 1)
        conv_out = self.relu(self.conv1(embedded))
        pooled = torch.max(conv_out, 2)[0]
        return self.fc1(pooled)

# โหลด BERT model
bert_model = BertModel.from_pretrained("bert-base-uncased")
model = HybridClassifier(bert_model, num_classes=2)
