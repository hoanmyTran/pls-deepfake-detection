import pytorch_lightning as pl
from data.audio import make_loader

class PLSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _loader(self, split):
        c = self.config
        ds = c['dataset']
        is_test = split == 'test'
        protocol = c['evaluation']['protocol_path'] if is_test else ds[split]['protocol_path']
        batch_size = c['evaluation']['batch_size'] if is_test else c['training']['batch_size']
        # A user-supplied `dataset.transform: true` previously meant repeat padding.
        fixed_length = ds.get('fixed_length', bool(ds.get('transform')))
        workers = c['evaluation'].get('num_workers', ds.get('num_workers', 0)) if is_test else ds.get('num_workers', 0)
        return make_loader(protocol, batch_size, num_workers=workers,
                           shuffle=split == 'train', sample_rate=ds.get('sampling_rate', 16000),
                           max_len=ds.get('max_len', 64600), fixed_length=fixed_length,
                           evaluate=is_test, rawboost=c.get('rawboost') if split == 'train' else None)

    def train_dataloader(self):
        return self._loader('train')

    def val_dataloader(self):
        return self._loader('val')

    def test_dataloader(self):
        return self._loader('test')
