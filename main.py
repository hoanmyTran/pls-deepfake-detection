import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import yaml

from data.lightning_data import PLSDataModule


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def extract_model_state(payload):
    """Extract plain nn.Module weights from old/new Lightning or plain checkpoints."""
    state = payload.get('state_dict', payload)
    if any(key.startswith('model.') for key in state):
        state = {key.removeprefix('model.'): value for key, value in state.items() if key.startswith('model.')}
    else:
        state = {key: value for key, value in state.items() if key.startswith(('feature_extractor.', 'classifier.'))}
    if not state:
        raise ValueError('Checkpoint contains no classifier/backbone weights')
    return state


def load_model_weights(model, checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(extract_model_state(payload), strict=True)


def make_trainer(cfg, *, evaluation=False):
    compute = cfg['compute']
    options = dict(accelerator=compute.get('accelerator', 'auto'),
                   devices=compute.get('device', 1), strategy=compute.get('strategy', 'auto'),
                   enable_model_summary=compute.get('enable_model_summary', True) and not evaluation)
    if evaluation:
        return pl.Trainer(logger=False, enable_checkpointing=False, **options)
    training = cfg['training']
    c = training.get('model_checkpoint', {})
    tag = cfg['model'].get('name', 'pls')
    checkpoint = ModelCheckpoint(
        dirpath=str(Path('results') / tag), monitor=c.get('ckpt_monitor', 'val_loss'),
        mode=c.get('ckpt_mode', 'min'), save_top_k=c.get('ckpt_save_top', 3),
        filename='epoch={epoch:02d}-val_loss={val_loss:.5f}-val_accuracy={val_accuracy:.4f}',
        auto_insert_metric_name=False, every_n_epochs=1, save_on_train_epoch_end=False,
    )
    callbacks = [checkpoint]
    es = training.get('early_stopping', {})
    if es.get('enabled', True):
        callbacks.append(EarlyStopping(monitor=es.get('es_monitor', 'val_loss'),
                                       mode=es.get('es_mode', 'min'), patience=es.get('patience', 3)))
    return pl.Trainer(max_epochs=training['num_epochs'], callbacks=callbacks,
                      check_val_every_n_epoch=1, val_check_interval=1.0, **options)


def run(cfg, *, evaluate=False, checkpoint=None):
    pl.seed_everything(cfg['training'].get('seed', 1234), workers=True)
    from trainers.lightning_module import LightningPLS
    model = LightningPLS(cfg)
    data = PLSDataModule(cfg)
    if evaluate:
        if not checkpoint:
            raise ValueError('Evaluation requires a --checkpoint')
        load_model_weights(model.model, checkpoint)
        make_trainer(cfg, evaluation=True).test(model, datamodule=data)
        return
    # .ckpt resumes optimizer, LR scheduler and epoch state; .pt initializes weights only.
    resume = None
    if checkpoint:
        if checkpoint.endswith('.ckpt'):
            resume = checkpoint
        else:
            load_model_weights(model.model, checkpoint)
    trainer = make_trainer(cfg)
    trainer.fit(model, datamodule=data, ckpt_path=resume)
    if trainer.is_global_zero:
        dest = Path('results') / cfg['model'].get('name', 'pls') / 'model_weights.pt'
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.model.state_dict(), dest)
        print(f'Exported plain PyTorch model weights: {dest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--protocol_path')
    parser.add_argument('--output_score_file')
    parser.add_argument('--device', type=int, nargs='+')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.protocol_path:
        cfg['evaluation']['protocol_path'] = args.protocol_path
    if args.output_score_file:
        cfg['evaluation']['output_score_file'] = args.output_score_file
    if args.device is not None:
        cfg['compute']['device'] = args.device
    run(cfg, evaluate=args.eval or cfg['evaluation'].get('eval', False),
        checkpoint=args.checkpoint or cfg['model'].get('checkpoint'))


if __name__ == '__main__':
    main()
