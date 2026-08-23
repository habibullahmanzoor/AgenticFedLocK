from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    'emnist_balanced': {
        'url': 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip',
        'archive_name': 'emnist_gzip.zip',
    },
    'femnist': {
        'url': 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz',
        'archive_name': 'femnist.tar.gz',
    },
    'cifar10': {
        'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        'archive_name': 'cifar-10-python.tar.gz',
    },
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36',
            'Accept': '*/*',
        },
    )
    with urllib.request.urlopen(request) as response, destination.open('wb') as handle:
        shutil.copyfileobj(response, handle)


def _prepare_emnist_balanced(data_root: Path, archive_path: Path) -> None:
    target_dir = data_root / 'emnist-balanced'
    target_dir.mkdir(parents=True, exist_ok=True)
    required = {
        'emnist-balanced-train-images-idx3-ubyte.gz',
        'emnist-balanced-train-labels-idx1-ubyte.gz',
        'emnist-balanced-test-images-idx3-ubyte.gz',
        'emnist-balanced-test-labels-idx1-ubyte.gz',
    }
    if all((target_dir / name).exists() for name in required):
        return
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            basename = Path(member).name
            if basename in required:
                with archive.open(member) as source, (target_dir / basename).open('wb') as dest:
                    shutil.copyfileobj(source, dest)
    missing = [name for name in required if not (target_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f'EMNIST Balanced extraction incomplete, missing: {", ".join(missing)}')


def _prepare_cifar10(data_root: Path, archive_path: Path) -> None:
    target_dir = data_root / 'cifar-10-batches-py'
    if target_dir.exists() and (target_dir / 'test_batch').exists():
        return
    with tarfile.open(archive_path, 'r:gz') as archive:
        archive.extractall(data_root)
    if not target_dir.exists():
        raise FileNotFoundError('CIFAR-10 extraction did not produce data/cifar-10-batches-py')


def _prepare_femnist(data_root: Path, archive_path: Path) -> None:
    processed_dir = data_root / 'FEMNIST' / 'processed'
    if (processed_dir / 'femnist_train.pt').exists() and (processed_dir / 'femnist_test.pt').exists():
        return
    processed_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, 'r:gz') as archive:
        archive.extractall(data_root)
    candidate_roots = [
        data_root / 'FEMNIST' / 'processed',
        data_root / 'femnist' / 'processed',
        data_root / 'FEMNIST',
        data_root / 'femnist',
        data_root,
    ]
    required = ['femnist_train.pt', 'femnist_test.pt']
    for required_name in required:
        if (processed_dir / required_name).exists():
            continue
        for root in candidate_roots:
            candidate = root / required_name
            if candidate.exists():
                if candidate.resolve() != (processed_dir / required_name).resolve():
                    shutil.move(str(candidate), str(processed_dir / required_name))
                break
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f'FEMNIST extraction incomplete, missing: {", ".join(missing)}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download and extract local datasets for AgenticFedLock benchmarks.')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS.keys()), default=['emnist_balanced', 'femnist', 'cifar10'])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloads_dir = args.data_root / '_downloads'
    downloads_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        spec = DATASETS[dataset]
        archive_path = downloads_dir / spec['archive_name']
        _download(spec['url'], archive_path)
        if dataset == 'emnist_balanced':
            _prepare_emnist_balanced(args.data_root, archive_path)
        elif dataset == 'femnist':
            _prepare_femnist(args.data_root, archive_path)
        elif dataset == 'cifar10':
            _prepare_cifar10(args.data_root, archive_path)
        else:
            raise ValueError(f'Unsupported dataset {dataset}')
        print(f'Prepared {dataset}')


if __name__ == '__main__':
    main()
