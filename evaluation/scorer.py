import csv
from pathlib import Path

class LayerScoreWriter:
    def __init__(self, output_dir, n_layers, flush_every=100):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        if n_layers < 1 or flush_every < 1:
            raise ValueError('n_layers and flush_every must be positive')
        self.suffixes = ['fp'] + [f'b{i}' for i in range(n_layers - 1)]
        self.buffers = {key: [] for key in self.suffixes}
        self.flush_every = int(flush_every)
        self.batches = 0
        for suffix in self.suffixes:
            with self.path(suffix).open('w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['ID', 'Score', 'Label'])

    def path(self, suffix):
        return self.root / f'score_{suffix}.csv'

    def write_batch(self, logits_list, paths, labels):
        if len(logits_list) != len(self.suffixes):
            raise ValueError(f'Expected {len(self.suffixes)} layers, got {len(logits_list)}')
        ids = [Path(str(p)).name for p in paths]
        y = labels.detach().cpu().tolist() if hasattr(labels, 'detach') else list(labels)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate basenames in one batch; use unique audio IDs')
        for suffix, logits in zip(self.suffixes, logits_list):
            pairs = logits.detach().cpu().tolist()
            if len(pairs) != len(ids) or len(y) != len(ids):
                raise ValueError('IDs, logits and labels length mismatch')
            self.buffers[suffix].extend((name, f'[{pair[0]}, {pair[1]}]', int(label)) for name, pair, label in zip(ids, pairs, y))
        self.batches += 1
        if self.batches % self.flush_every == 0:
            self.flush()

    def flush(self):
        for suffix, rows in self.buffers.items():
            if rows:
                with self.path(suffix).open('a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(rows)
                rows.clear()

    def close(self):
        self.flush()
