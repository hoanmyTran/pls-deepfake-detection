import argparse
import copy
from pathlib import Path
import re

import pytorch_lightning as pl
import yaml
from data.lightning_data import PLSDataModule
from main import load_config, load_model_weights, make_trainer
from compute_eer_all_blocks import summarize


def read_dataset_manifest(path):
    with open(path, encoding='utf-8') as stream:
        datasets = (yaml.safe_load(stream) or {}).get('datasets', {})
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError('Manifest must contain a nonempty datasets mapping')
    for name, entry in datasets.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', str(name)):
            raise ValueError(f'Unsafe dataset name: {name}')
        if not isinstance(entry, dict) or not entry.get('protocol_path'):
            raise ValueError(f'Missing protocol_path for {name}')
    return datasets


def pick_datasets(manifest, requested=None, all_datasets=False):
    if requested:
        unknown = set(requested) - set(manifest)
        if unknown:
            raise ValueError(f'Unknown datasets: {sorted(unknown)}')
        return [name for name in manifest if name in requested]
    return [name for name, entry in manifest.items() if all_datasets or entry.get('enabled', True)]


def run(cfg, definitions, selected, checkpoint, output_root, *, device_id=None, aggregate=True):
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')
    if not selected:
        raise ValueError('No datasets selected')
    for name in selected:
        if not Path(definitions[name]['protocol_path']).expanduser().is_file():
            raise FileNotFoundError(f'{name}: missing protocol: {definitions[name]["protocol_path"]}')
    pl.seed_everything(cfg['training'].get('seed', 1234), workers=True)
    from trainers.lightning_module import LightningPLS
    wrapper = LightningPLS(cfg)
    load_model_weights(wrapper.model, checkpoint)
    settings = copy.deepcopy(cfg)
    if device_id is not None:
        settings['compute']['device'] = [device_id]
    trainer = make_trainer(settings, evaluation=True)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in selected:
        per_dataset = copy.deepcopy(settings)
        per_dataset['evaluation']['protocol_path'] = str(Path(definitions[name]['protocol_path']).expanduser())
        per_dataset['evaluation']['output_score_file'] = str(output_root / name)
        wrapper.config = per_dataset
        print(f'Evaluating {name}', flush=True)
        trainer.test(wrapper, datamodule=PLSDataModule(per_dataset), verbose=False)
    if aggregate:
        summarize(output_root, datasets=selected, output_csv=output_root / 'summary.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--datasets-file')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', type=int)
    parser.add_argument('--output-root')
    parser.add_argument('--no-summary', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = read_dataset_manifest(args.datasets_file or cfg['evaluation'].get('datasets_file', 'datasets.yaml'))
    selected = pick_datasets(manifest, args.datasets, args.all)
    run(cfg, manifest, selected, args.checkpoint,
        args.output_root or cfg['evaluation'].get('output_root', 'scores/pls_xlsr300m'),
        device_id=args.device, aggregate=not args.no_summary)


if __name__ == '__main__':
    main()
