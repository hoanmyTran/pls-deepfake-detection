import torch
from torch import nn
from transformers import AutoModel


class PLSClassifier(nn.Module):
    def __init__(self, pretrained_model: str = 'facebook/wav2vec2-xls-r-300m'):
        super().__init__()
        self.feature_extractor = AutoModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.feature_extractor.config.hidden_size, 2)

    def forward(self, audio: torch.Tensor, return_all_layers: bool = False):
        states = self.feature_extractor(audio, output_hidden_states=True).hidden_states
        logits = [self.classifier(state.mean(dim=1)) for state in states]
        return logits if return_all_layers else logits[-1]
