import torch
import torch.nn as nn
from models import ElectraSequenceClassifier, BertSequenceClassifier, RobertaSequenceClassifier, XlnetSequenceClassifier

class Attack_Embedding():
    '''
    Structure that perturbs encoded sentence embedding in attack direction
    '''
    def __init__(self, model, name):
        self.model = model.eval()
        self.name = name # e.g. electra

    def attack(self, input_ids, attention_mask, attack_direction):
        if self.name == 'electra':
            return self._electra_attack(input_ids, attention_mask, attack_direction)
        else:
            raise Exception("Model type not yet supported")

    def _electra_attack(self, input_ids, attention_mask, attack_direction):
        all_layers_hidden_states = self.model.electra(input_ids, attention_mask)
        final_layer = all_layers_hidden_states[0]
        sentence_embedding = final_layer[:,0,:]

        # Attack embeddings
        attacked_embedding = sentence_embedding + attack_direction

        # Pass through remainder of classifier
        x = self.model.classifier.dropout(attacked_embedding)
        x = self.model.classifier.layer(x)
        m = nn.GELU()
        x = m(x) # gelu used by electra authors
        x = self.model.classifier.dropout(x)
        logits = self.model.classifier.out_proj(x)
        return logits
