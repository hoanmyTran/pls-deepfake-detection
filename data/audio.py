from pathlib import Path
import csv
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def read_protocol(path):
    records = []
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        for line, row in enumerate(reader, 1):
            if not row or not any(v.strip() for v in row):
                continue
            if len(row) < 2:
                raise ValueError(f'{path}:{line}: expected file_name,label')
            filename, label = row[0].strip(), row[1].strip().lower()
            if line == 1 and filename.lower() == 'file_name' and label == 'label':
                continue
            if not filename or label not in {'spoof', 'bonafide', '0', '1'}:
                raise ValueError(f'{path}:{line}: invalid audio path or label: {row[:2]}')
            records.append((filename, 1 if label in ('bonafide', '1') else 0))
    if not records:
        raise ValueError(f'No audio records found in {path}')
    return records


def repeat_pad(audio, max_len):
    x = np.asarray(audio).reshape(-1)
    if max_len <= 0 or len(x) == 0:
        raise ValueError('max_len must be positive and audio must be nonempty')
    if len(x) >= max_len:
        return x[:max_len].copy()
    return np.tile(x, (max_len + len(x) - 1) // len(x))[:max_len].copy()


class AudioDataset(Dataset):
    def __init__(self, protocol, *, sample_rate=16000, max_len=64600,
                 fixed_length=False, rawboost=None, return_metadata=False):
        self.records = read_protocol(protocol)
        self.sample_rate = int(sample_rate)
        self.max_len = int(max_len)
        self.fixed_length = bool(fixed_length)
        self.rawboost = rawboost if rawboost and int(rawboost.get('algo', 0)) else None
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        audio, _ = librosa.load(path, sr=self.sample_rate)
        if not len(audio):
            raise ValueError(f'Empty audio: {path}')
        if self.rawboost is not None:
            from data.rawboost_adapter import apply_rawboost
            audio = apply_rawboost(audio, self.rawboost, self.sample_rate)
        if self.fixed_length:
            audio = repeat_pad(audio, self.max_len)
        tensor = torch.as_tensor(np.asarray(audio, dtype=np.float32).copy())
        return (tensor, path, label) if self.return_metadata else (tensor, label)


def collate_train(batch):
    waveforms, labels = zip(*batch)
    return pad_sequence(waveforms, batch_first=True, padding_value=0), torch.tensor(labels, dtype=torch.long)


def collate_eval(batch):
    waveforms, paths, labels = zip(*batch)
    return (pad_sequence(waveforms, batch_first=True, padding_value=0),
            tuple(paths), torch.tensor(labels, dtype=torch.long))


def make_loader(protocol, batch_size, *, num_workers=0, shuffle=False,
                sample_rate=16000, max_len=64600, fixed_length=False,
                evaluate=False, rawboost=None, pin_memory=None, prefetch_factor=2):
    dataset = AudioDataset(protocol, sample_rate=sample_rate, max_len=max_len,
                           fixed_length=fixed_length, return_metadata=evaluate,
                           rawboost=rawboost if not evaluate else None)
    opts = dict(batch_size=int(batch_size), num_workers=int(num_workers), shuffle=shuffle,
                collate_fn=collate_eval if evaluate else collate_train,
                pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory))
    if num_workers > 0:
        opts.update(persistent_workers=True, prefetch_factor=int(prefetch_factor))
    return DataLoader(dataset, **opts)
