import pytorch_lightning as pl
import torch
from models.pls_classifier import PLSClassifier
from losses.progressive_loss import ProgressiveLayerwiseLoss
from evaluation.scorer import LayerScoreWriter


class LightningPLS(pl.LightningModule):
    def __init__(self, config, model=None):
        super().__init__()
        self.config = config
        cfg = config['model']
        self.model = model if model is not None else PLSClassifier(cfg['pretrained_model'])
        self.criterion = ProgressiveLayerwiseLoss(cfg.get('max_power', 1.5), cfg.get('k_power', 0.05), cfg.get('class_weights', [0.1, 0.9]))
        self.writer = None

    def forward(self, audio, return_all_layers=False):
        return self.model(audio, return_all_layers=return_all_layers)

    def _step(self, batch, stage):
        x, y = batch
        logits = self.model(x, return_all_layers=True)
        loss = self.criterion(logits, y, self.current_epoch)
        accuracy = (logits[-1].argmax(-1) == y).float().mean()
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y), sync_dist=True)
        self.log(f'{stage}_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def configure_optimizers(self):
        cfg = self.config['training']
        return torch.optim.Adam(self.model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg.get('weight_decay', 0))

    def on_test_start(self):
        cfg = self.config['evaluation']
        self.writer = LayerScoreWriter(cfg['output_score_file'], len(self.model.feature_extractor.encoder.layers) + 1,
                                       flush_every=cfg.get('score_flush_batches', 100))

    def test_step(self, batch, batch_idx):
        x, paths, y = batch
        self.writer.write_batch(self.model(x, return_all_layers=True), paths, y)

    def on_test_end(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
