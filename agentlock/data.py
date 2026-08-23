from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import json
import os
from pathlib import Path
import pickle
import random
import struct
import tempfile

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from agentlock.config import SimulationConfig
from agentlock.model import ModelSpec, build_model_spec


@dataclass
class ClientProfile:
    client_id: int
    sample_count: int
    label_histogram: list[int]
    observed_labels: list[int]
    bandwidth_score: float
    rarity_score: float
    is_malicious: bool


@dataclass
class TaskData:
    dataset_name: str
    train_dataset: TensorDataset
    probe_dataset: TensorDataset
    test_dataset: TensorDataset
    client_indices: dict[int, list[int]]
    client_profiles: dict[int, ClientProfile]
    input_dim: int
    feature_shape: tuple[int, ...]
    num_classes: int
    rare_labels: tuple[int, ...]
    backdoor_target: int
    trigger_kind: str
    trigger_indices: tuple[int, ...]
    trigger_value: float
    model_spec: ModelSpec
    image_trigger_variant: str = "solid_patch"


@dataclass(frozen=True)
class DatasetBundle:
    features: np.ndarray
    labels: np.ndarray
    trigger_kind: str
    test_features: np.ndarray | None = None
    test_labels: np.ndarray | None = None
    client_indices: dict[int, list[int]] | None = None


_DATASET_CHOICES = (
    'sklearn_digits',
    'sklearn_wine',
    'sklearn_breast_cancer',
    'mnist_local',
    'fashion_mnist_local',
    'emnist_balanced_local',
    'femnist_local',
    'cifar10_local',
)


def available_datasets() -> tuple[str, ...]:
    return _DATASET_CHOICES


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_client_size: int,
    seed: int,
) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    while True:
        per_client: dict[int, list[int]] = {client_id: [] for client_id in range(num_clients)}
        for label in classes:
            label_indices = np.where(labels == label)[0]
            rng.shuffle(label_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            boundaries = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
            chunks = np.split(label_indices, boundaries)
            for client_id, chunk in enumerate(chunks):
                per_client[client_id].extend(chunk.tolist())
        sizes = [len(indices) for indices in per_client.values()]
        if min(sizes) >= min_client_size:
            for indices in per_client.values():
                rng.shuffle(indices)
            return per_client


def _stratified_data_subset(
    features: np.ndarray,
    labels: np.ndarray,
    data_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a reproducible class-stratified subset of a training pool."""
    if data_fraction >= 1.0:
        return features, labels
    selected_x, _, selected_y, _ = train_test_split(
        features,
        labels,
        train_size=data_fraction,
        random_state=seed,
        stratify=labels,
    )
    return selected_x, selected_y


def _tensor_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    # Share the already-owned split buffers instead of creating another full
    # dataset-sized tensor copy. Compact uint8 images are normalized per batch.
    x_tensor = torch.from_numpy(np.ascontiguousarray(features))
    y_tensor = torch.from_numpy(np.ascontiguousarray(labels.astype(np.int64, copy=False)))
    return TensorDataset(x_tensor, y_tensor)


def _find_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_binary(path: Path) -> bytes:
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as handle:
            return handle.read()
    return path.read_bytes()


def _load_idx_images(path: Path) -> np.ndarray:
    raw = _read_binary(path)
    magic, count, rows, cols = struct.unpack_from('>IIII', raw, 0)
    if magic != 2051:
        raise ValueError(f'Unexpected IDX image magic {magic} in {path}')
    data = np.frombuffer(raw, dtype=np.uint8, offset=16)
    images = data.reshape(count, 1, rows, cols).astype(np.float32) / 255.0
    return images


def _load_idx_labels(path: Path) -> np.ndarray:
    raw = _read_binary(path)
    magic, count = struct.unpack_from('>II', raw, 0)
    if magic != 2049:
        raise ValueError(f'Unexpected IDX label magic {magic} in {path}')
    data = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return data[:count].astype(np.int64)


def _fix_emnist_orientation(images: np.ndarray) -> np.ndarray:
    # EMNIST IDX files are transposed relative to MNIST; transpose + horizontal flip restores upright glyphs.
    images = np.transpose(images, (0, 1, 3, 2))
    return np.flip(images, axis=3).copy()


def _load_local_mnist_bundle(dataset_name: str, data_root: Path) -> DatasetBundle:
    if dataset_name == 'mnist_local':
        folder_name = 'mnist'
        train_image_names = ['train-images-idx3-ubyte', 'train-images-idx3-ubyte.gz']
        train_label_names = ['train-labels-idx1-ubyte', 'train-labels-idx1-ubyte.gz']
        test_image_names = ['t10k-images-idx3-ubyte', 't10k-images-idx3-ubyte.gz']
        test_label_names = ['t10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte.gz']
        needs_emnist_fix = False
    elif dataset_name == 'fashion_mnist_local':
        folder_name = 'fashion-mnist'
        train_image_names = ['train-images-idx3-ubyte', 'train-images-idx3-ubyte.gz']
        train_label_names = ['train-labels-idx1-ubyte', 'train-labels-idx1-ubyte.gz']
        test_image_names = ['t10k-images-idx3-ubyte', 't10k-images-idx3-ubyte.gz']
        test_label_names = ['t10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte.gz']
        needs_emnist_fix = False
    else:
        folder_name = 'emnist-balanced'
        train_image_names = [
            'emnist-balanced-train-images-idx3-ubyte',
            'emnist-balanced-train-images-idx3-ubyte.gz',
            'train-images-idx3-ubyte',
            'train-images-idx3-ubyte.gz',
        ]
        train_label_names = [
            'emnist-balanced-train-labels-idx1-ubyte',
            'emnist-balanced-train-labels-idx1-ubyte.gz',
            'train-labels-idx1-ubyte',
            'train-labels-idx1-ubyte.gz',
        ]
        test_image_names = [
            'emnist-balanced-test-images-idx3-ubyte',
            'emnist-balanced-test-images-idx3-ubyte.gz',
            't10k-images-idx3-ubyte',
            't10k-images-idx3-ubyte.gz',
        ]
        test_label_names = [
            'emnist-balanced-test-labels-idx1-ubyte',
            'emnist-balanced-test-labels-idx1-ubyte.gz',
            't10k-labels-idx1-ubyte',
            't10k-labels-idx1-ubyte.gz',
        ]
        needs_emnist_fix = True

    dataset_root = data_root / folder_name
    train_images = _find_existing_path([dataset_root / name for name in train_image_names])
    train_labels = _find_existing_path([dataset_root / name for name in train_label_names])
    test_images = _find_existing_path([dataset_root / name for name in test_image_names])
    test_labels = _find_existing_path([dataset_root / name for name in test_label_names])
    missing = [
        name
        for name, value in {
            'train images': train_images,
            'train labels': train_labels,
            'test images': test_images,
            'test labels': test_labels,
        }.items()
        if value is None
    ]
    if missing:
        missing_text = ', '.join(missing)
        raise FileNotFoundError(
            f'Missing {missing_text} for {dataset_name}. Expected local IDX files under {dataset_root}.',
        )
    train_x = _load_idx_images(train_images)
    test_x = _load_idx_images(test_images)
    if needs_emnist_fix:
        train_x = _fix_emnist_orientation(train_x)
        test_x = _fix_emnist_orientation(test_x)
    return DatasetBundle(
        features=train_x,
        labels=_load_idx_labels(train_labels),
        test_features=test_x,
        test_labels=_load_idx_labels(test_labels),
        trigger_kind='image_patch',
    )


def _load_cifar10_python_batches(batch_dir: Path) -> DatasetBundle:
    train_batches = [batch_dir / f'data_batch_{index}' for index in range(1, 6)]
    test_batch = batch_dir / 'test_batch'
    missing = [str(path.name) for path in train_batches + [test_batch] if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing CIFAR-10 python batch files under {batch_dir}: {", ".join(missing)}')

    def _read_pickle(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open('rb') as handle:
            payload = pickle.load(handle, encoding='bytes')
        data = payload[b'data'] if b'data' in payload else payload['data']
        labels = payload.get(b'labels') if b'labels' in payload else payload.get('labels')
        if labels is None:
            labels = payload.get(b'fine_labels') if b'fine_labels' in payload else payload.get('fine_labels')
        # Keep CIFAR-10 compact until a minibatch is moved to the model.
        images = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32)
        return images, np.asarray(labels, dtype=np.int64)

    train_images, train_labels = zip(*[_read_pickle(path) for path in train_batches])
    test_images, test_labels = _read_pickle(test_batch)
    return DatasetBundle(
        features=np.concatenate(train_images, axis=0),
        labels=np.concatenate(train_labels, axis=0),
        test_features=test_images,
        test_labels=test_labels,
        trigger_kind='image_patch',
    )


def _load_cifar10_binary_batches(batch_dir: Path) -> DatasetBundle:
    train_batches = [batch_dir / f'data_batch_{index}.bin' for index in range(1, 6)]
    test_batch = batch_dir / 'test_batch.bin'
    missing = [str(path.name) for path in train_batches + [test_batch] if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing CIFAR-10 binary batch files under {batch_dir}: {", ".join(missing)}')

    def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
        raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        rows = raw.reshape(-1, 3073)
        labels = rows[:, 0].astype(np.int64)
        images = rows[:, 1:].reshape(-1, 3, 32, 32).copy()
        return images, labels

    train_images, train_labels = zip(*[_read_batch(path) for path in train_batches])
    test_images, test_labels = _read_batch(test_batch)
    return DatasetBundle(
        features=np.concatenate(train_images, axis=0),
        labels=np.concatenate(train_labels, axis=0),
        test_features=test_images,
        test_labels=test_labels,
        trigger_kind='image_patch',
    )


def _load_cifar10_bundle(data_root: Path) -> DatasetBundle:
    python_dir = data_root / 'cifar-10-batches-py'
    binary_dir = data_root / 'cifar-10-batches-bin'
    if python_dir.exists():
        return _load_cifar10_python_batches(python_dir)
    if binary_dir.exists():
        return _load_cifar10_binary_batches(binary_dir)
    raise FileNotFoundError(
        'Missing CIFAR-10 data. Expected extracted python batches under data/cifar-10-batches-py or binary batches under data/cifar-10-batches-bin.',
    )


def _normalize_leaf_images(values: list[object]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == 28 * 28:
        array = array.reshape(-1, 1, 28, 28)
    elif array.ndim == 3:
        array = array[:, None, :, :]
    elif array.ndim == 4 and array.shape[1] != 1:
        raise ValueError(f'Unexpected FEMNIST image shape: {array.shape}')
    return array / 255.0


def _load_leaf_split(split_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if not split_dir.exists():
        return []
    users: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path in sorted(split_dir.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        user_data = payload.get('user_data', {})
        users_in_file = payload.get('users', sorted(user_data.keys()))
        for user in users_in_file:
            if user not in user_data:
                continue
            entry = user_data[user]
            x_values = entry.get('x', [])
            y_values = entry.get('y', [])
            if not x_values:
                continue
            users.append((
                str(user),
                _normalize_leaf_images(x_values),
                np.asarray(y_values, dtype=np.int64),
            ))
    return users


def _torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _normalize_femnist_pt_images(values: object) -> np.ndarray:
    # Avoid materializing a float64 copy of the full FEMNIST corpus (~650k
    # images): cast directly to float32, then collapse to compact uint8 and
    # defer [0,1] normalization to per-minibatch model input, matching the
    # CIFAR-10 loading path.
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        array = np.asarray(values, dtype=np.float32)
    if array.max() <= 1.0:
        array *= 255.0
    np.clip(array, 0.0, 255.0, out=array)
    array = array.astype(np.uint8)
    if array.ndim == 2 and array.shape[1] == 28 * 28:
        array = array.reshape(-1, 1, 28, 28)
    elif array.ndim == 3:
        array = array[:, None, :, :]
    elif array.ndim == 4 and array.shape[1] not in {1, 3}:
        array = np.transpose(array, (0, 3, 1, 2))
    elif array.ndim != 4:
        raise ValueError(f'Unexpected FEMNIST tensor shape: {array.shape}')
    return array


def _client_indices_from_user_sequence(users: list[object]) -> dict[int, list[int]]:
    buckets: dict[str, list[int]] = {}
    ordered_users: list[str] = []
    for index, user in enumerate(users):
        key = str(user)
        if key not in buckets:
            buckets[key] = []
            ordered_users.append(key)
        buckets[key].append(index)
    return {client_id: buckets[user] for client_id, user in enumerate(ordered_users)}


_FEMNIST_CACHE_VERSION = 'v1'


def _femnist_cache_paths(root: Path) -> tuple[Path, Path]:
    cache_dir = root / '_cache'
    stem = f'femnist_bundle_{_FEMNIST_CACHE_VERSION}'
    return cache_dir / f'{stem}.npz', cache_dir / f'{stem}_client_indices.json'


def _read_femnist_cache(root: Path) -> DatasetBundle | None:
    cache_npz, cache_json = _femnist_cache_paths(root)
    if not cache_npz.exists() or not cache_json.exists():
        return None
    try:
        with np.load(cache_npz) as cached:
            train_x = cached['train_x']
            train_y = cached['train_y']
            test_x = cached['test_x']
            test_y = cached['test_y']
        client_indices = {
            int(key): value for key, value in json.loads(cache_json.read_text(encoding='utf-8')).items()
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        # A partial or stale cache falls back to re-parsing the raw .pt files
        # rather than failing the run.
        return None
    return DatasetBundle(
        features=train_x,
        labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        trigger_kind='image_patch',
        client_indices=client_indices,
    )


def _write_femnist_cache(
    root: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    client_indices: dict[int, list[int]],
) -> None:
    cache_npz, cache_json = _femnist_cache_paths(root)
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    # Write to a per-process temp file and atomically replace, so a second
    # concurrent loader (or a crash mid-write) never observes a partial cache.
    fd, tmp_npz_name = tempfile.mkstemp(dir=cache_npz.parent, suffix='.npz.tmp')
    tmp_npz = Path(tmp_npz_name)
    try:
        # Pass an open handle rather than a path: np.savez silently appends
        # '.npz' to string paths that lack the suffix, which would otherwise
        # write the real data to a different file than the one we replace.
        with os.fdopen(fd, 'wb') as handle:
            np.savez(handle, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
        os.replace(tmp_npz, cache_npz)
    finally:
        tmp_npz.unlink(missing_ok=True)
    tmp_json_name = tempfile.mktemp(dir=cache_json.parent, suffix='.json.tmp')
    tmp_json = Path(tmp_json_name)
    tmp_json.write_text(
        json.dumps({str(key): value for key, value in client_indices.items()}),
        encoding='utf-8',
    )
    os.replace(tmp_json, cache_json)


def _load_femnist_pt_bundle(data_root: Path) -> DatasetBundle | None:
    candidate_roots = [
        data_root / 'FEMNIST' / 'processed',
        data_root / 'femnist' / 'processed',
        data_root / 'FEMNIST',
        data_root / 'femnist',
    ]
    for root in candidate_roots:
        train_path = root / 'femnist_train.pt'
        test_path = root / 'femnist_test.pt'
        if not train_path.exists() or not test_path.exists():
            continue
        cached_bundle = _read_femnist_cache(root)
        if cached_bundle is not None:
            return cached_bundle
        train_payload = _torch_load_compat(train_path)
        test_payload = _torch_load_compat(test_path)
        if not isinstance(train_payload, (list, tuple)) or len(train_payload) < 3:
            raise ValueError(f'Unexpected FEMNIST train payload format in {train_path}')
        if not isinstance(test_payload, (list, tuple)) or len(test_payload) < 3:
            raise ValueError(f'Unexpected FEMNIST test payload format in {test_path}')
        train_x = _normalize_femnist_pt_images(train_payload[0])
        train_y = np.asarray(train_payload[1], dtype=np.int64)
        train_users = list(train_payload[2])
        test_x = _normalize_femnist_pt_images(test_payload[0])
        test_y = np.asarray(test_payload[1], dtype=np.int64)
        client_indices = _client_indices_from_user_sequence(train_users)
        _write_femnist_cache(root, train_x, train_y, test_x, test_y, client_indices)
        return DatasetBundle(
            features=train_x,
            labels=train_y,
            test_features=test_x,
            test_labels=test_y,
            trigger_kind='image_patch',
            client_indices=client_indices,
        )
    return None


def _combine_user_samples(users: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    client_indices: dict[int, list[int]] = {}
    cursor = 0
    for client_id, (_user, features, labels) in enumerate(users):
        count = int(labels.shape[0])
        all_x.append(features)
        all_y.append(labels)
        client_indices[client_id] = list(range(cursor, cursor + count))
        cursor += count
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0), client_indices


def _split_leaf_users(users: list[tuple[str, np.ndarray, np.ndarray]], seed: int) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], list[tuple[str, np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    train_users: list[tuple[str, np.ndarray, np.ndarray]] = []
    test_users: list[tuple[str, np.ndarray, np.ndarray]] = []
    for user, features, labels in users:
        count = int(labels.shape[0])
        if count < 5:
            train_users.append((user, features, labels))
            continue
        perm = rng.permutation(count)
        test_count = max(1, int(round(count * 0.2)))
        test_idx = perm[:test_count]
        train_idx = perm[test_count:]
        if train_idx.size == 0:
            train_idx = test_idx[:1]
            test_idx = test_idx[1:]
        train_users.append((user, features[train_idx], labels[train_idx]))
        if test_idx.size > 0:
            test_users.append((user, features[test_idx], labels[test_idx]))
    return train_users, test_users


def _load_femnist_bundle(data_root: Path) -> DatasetBundle:
    pt_bundle = _load_femnist_pt_bundle(data_root)
    if pt_bundle is not None:
        return pt_bundle

    femnist_root = data_root / 'femnist'
    train_users = _load_leaf_split(femnist_root / 'train')
    test_users = _load_leaf_split(femnist_root / 'test')
    if not train_users and not test_users:
        train_users = _load_leaf_split(femnist_root / 'data' / 'train')
        test_users = _load_leaf_split(femnist_root / 'data' / 'test')
    if not train_users and not test_users:
        all_users = _load_leaf_split(femnist_root / 'all_data')
        if not all_users:
            raise FileNotFoundError(
                'Missing FEMNIST data. Expected LEAF-style JSON under data/femnist/train + test, data/femnist/data/train + test, or data/femnist/all_data.',
            )
        train_users, test_users = _split_leaf_users(all_users, seed=0)
    if not train_users:
        raise FileNotFoundError('FEMNIST train split is empty or missing.')
    train_x, train_y, client_indices = _combine_user_samples(train_users)
    if test_users:
        test_x, test_y, _ = _combine_user_samples(test_users)
    else:
        test_x = None
        test_y = None
    return DatasetBundle(
        features=train_x,
        labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        trigger_kind='image_patch',
        client_indices=client_indices,
    )


def _load_dataset_bundle(dataset_name: str, data_root: Path) -> DatasetBundle:
    if dataset_name == 'sklearn_digits':
        digits = load_digits()
        features = digits.images.astype(np.float32) / 16.0
        features = features[:, None, :, :]
        labels = digits.target.astype(np.int64)
        return DatasetBundle(features=features, labels=labels, trigger_kind='image_patch')
    if dataset_name == 'sklearn_wine':
        wine = load_wine()
        return DatasetBundle(features=wine.data.astype(np.float32), labels=wine.target.astype(np.int64), trigger_kind='feature_stamp')
    if dataset_name == 'sklearn_breast_cancer':
        cancer = load_breast_cancer()
        return DatasetBundle(features=cancer.data.astype(np.float32), labels=cancer.target.astype(np.int64), trigger_kind='feature_stamp')
    if dataset_name in {'mnist_local', 'fashion_mnist_local', 'emnist_balanced_local'}:
        return _load_local_mnist_bundle(dataset_name, data_root)
    if dataset_name == 'femnist_local':
        return _load_femnist_bundle(data_root)
    if dataset_name == 'cifar10_local':
        return _load_cifar10_bundle(data_root)
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def _standardize_tabular(
    train_x: np.ndarray,
    probe_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x).astype(np.float32)
    probe_x = scaler.transform(probe_x).astype(np.float32)
    test_x = scaler.transform(test_x).astype(np.float32)
    return train_x, probe_x, test_x


def _resolve_rare_labels(labels: np.ndarray, configured: tuple[int, ...]) -> tuple[int, ...]:
    num_classes = int(labels.max()) + 1
    filtered = tuple(sorted({label for label in configured if 0 <= label < num_classes}))
    if filtered:
        return filtered
    label_hist = np.bincount(labels, minlength=num_classes)
    sorted_labels = np.argsort(label_hist)
    take = 2 if num_classes >= 6 else 1
    return tuple(int(label) for label in sorted_labels[:take])


def _tabular_trigger_indices(train_x: np.ndarray) -> tuple[int, ...]:
    flat = train_x.reshape(train_x.shape[0], -1)
    feature_count = flat.shape[1]
    trigger_size = 2 if feature_count <= 16 else 3
    variances = flat.var(axis=0)
    selected = np.argsort(variances)[:trigger_size]
    return tuple(int(index) for index in sorted(selected.tolist()))


def _build_client_profiles(
    train_y: np.ndarray,
    client_indices: dict[int, list[int]],
    num_classes: int,
    rare_labels: tuple[int, ...],
    bandwidth_levels: tuple[float, float, float],
    malicious_clients: set[int],
) -> dict[int, ClientProfile]:
    global_label_hist = np.bincount(train_y, minlength=num_classes).astype(np.float64)
    global_label_hist /= global_label_hist.sum()
    global_rare_mass = float(global_label_hist[list(rare_labels)].sum())

    bandwidth_cycle = list(bandwidth_levels)
    client_profiles: dict[int, ClientProfile] = {}
    for client_id, indices in client_indices.items():
        local_labels = train_y[indices]
        label_hist = np.bincount(local_labels, minlength=num_classes)
        observed_labels = np.flatnonzero(label_hist).tolist()
        rare_mass = float(label_hist[list(rare_labels)].sum() / max(1, len(indices)))
        rarity_score = (rare_mass - global_rare_mass) / max(1e-6, 1.0 - global_rare_mass)
        rarity_score = float(np.clip(0.5 + rarity_score, 0.0, 1.0))
        bandwidth_score = bandwidth_cycle[client_id % len(bandwidth_cycle)]
        client_profiles[client_id] = ClientProfile(
            client_id=client_id,
            sample_count=len(indices),
            label_histogram=label_hist.tolist(),
            observed_labels=observed_labels,
            bandwidth_score=bandwidth_score,
            rarity_score=rarity_score,
            is_malicious=client_id in malicious_clients,
        )
    return client_profiles


def _apply_rarity_information_mode(
    client_profiles: dict[int, ClientProfile],
    mode: str,
) -> dict[int, ClientProfile]:
    """Construct the controller-visible rarity prior for an ablation condition.

    ``probe_feedback`` intentionally exposes no client label-histogram-derived
    prior: every client starts neutral and can only acquire a rarity signal from
    observed selected-update probe effects. ``none`` also remains neutral after
    feedback. ``malicious_spoof`` is a declared-rarity stress condition in which
    adversarial clients report the maximum possible rarity prior.
    """
    if mode == "label_oracle":
        return client_profiles
    transformed: dict[int, ClientProfile] = {}
    for client_id, profile in client_profiles.items():
        if mode in {"probe_feedback", "none"}:
            rarity_score = 0.5
        elif mode == "malicious_spoof" and profile.is_malicious:
            rarity_score = 1.0
        else:
            rarity_score = profile.rarity_score
        transformed[client_id] = replace(profile, rarity_score=rarity_score)
    return transformed


def _select_predefined_clients(
    features: np.ndarray,
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
    num_clients: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    available_client_ids = sorted(client_indices.keys())
    if num_clients >= len(available_client_ids):
        selected_ids = available_client_ids
    else:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(np.asarray(available_client_ids), size=num_clients, replace=False)
        selected_ids = sorted(int(client_id) for client_id in sampled.tolist())

    selected_features: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []
    remapped_indices: dict[int, list[int]] = {}
    cursor = 0
    for new_client_id, old_client_id in enumerate(selected_ids):
        indices = client_indices[old_client_id]
        client_x = features[indices]
        client_y = labels[indices]
        selected_features.append(client_x)
        selected_labels.append(client_y)
        remapped_indices[new_client_id] = list(range(cursor, cursor + len(indices)))
        cursor += len(indices)
    return np.concatenate(selected_features, axis=0), np.concatenate(selected_labels, axis=0), remapped_indices


def _subsample_predefined_client_data(
    features: np.ndarray,
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
    data_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Retain a deterministic fraction within each predefined client population."""
    if data_fraction >= 1.0:
        return features, labels, client_indices

    rng = np.random.default_rng(seed)
    kept_original_indices: list[int] = []
    kept_by_client: dict[int, list[int]] = {}
    for client_id, indices in client_indices.items():
        client_array = np.asarray(indices, dtype=np.int64)
        take = max(1, int(round(client_array.size * data_fraction)))
        chosen = np.sort(rng.choice(client_array, size=take, replace=False))
        kept_by_client[client_id] = chosen.tolist()
        kept_original_indices.extend(chosen.tolist())

    kept_original = np.asarray(kept_original_indices, dtype=np.int64)
    remap = {int(original): new for new, original in enumerate(kept_original.tolist())}
    remapped_client_indices = {
        client_id: [remap[int(index)] for index in indices]
        for client_id, indices in kept_by_client.items()
    }
    return features[kept_original], labels[kept_original], remapped_client_indices


def _extract_probe_from_predefined_clients(
    features: np.ndarray,
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
    probe_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, list[int]]]:
    rng = np.random.default_rng(seed)
    train_features: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    probe_features: list[np.ndarray] = []
    probe_labels: list[np.ndarray] = []
    remapped_indices: dict[int, list[int]] = {}
    cursor = 0

    for client_id in sorted(client_indices.keys()):
        indices = np.asarray(client_indices[client_id], dtype=np.int64)
        perm = rng.permutation(indices)
        if perm.size <= 2:
            probe_take = 0
        else:
            probe_take = min(max(1, int(round(perm.size * probe_fraction))), perm.size - 2)
        probe_idx = perm[:probe_take]
        train_idx = perm[probe_take:]
        client_train_x = features[train_idx]
        client_train_y = labels[train_idx]
        train_features.append(client_train_x)
        train_labels.append(client_train_y)
        remapped_indices[client_id] = list(range(cursor, cursor + int(client_train_y.shape[0])))
        cursor += int(client_train_y.shape[0])
        if probe_idx.size:
            probe_features.append(features[probe_idx])
            probe_labels.append(labels[probe_idx])

    train_x = np.concatenate(train_features, axis=0)
    train_y = np.concatenate(train_labels, axis=0)
    if probe_features:
        probe_x = np.concatenate(probe_features, axis=0)
        probe_y = np.concatenate(probe_labels, axis=0)
    else:
        probe_x = train_x[:1].copy()
        probe_y = train_y[:1].copy()
    return train_x, probe_x, train_y, probe_y, remapped_indices


def build_task(config: SimulationConfig) -> TaskData:
    bundle = _load_dataset_bundle(config.dataset_name, config.data_root)
    trigger_kind = bundle.trigger_kind

    if bundle.client_indices is None:
        if bundle.test_features is None or bundle.test_labels is None:
            features = bundle.features
            labels = bundle.labels
            train_x, test_x, train_y, test_y = train_test_split(
                features,
                labels,
                test_size=config.test_fraction,
                random_state=config.seed,
                stratify=labels,
            )
        else:
            train_x = bundle.features
            train_y = bundle.labels
            test_x = bundle.test_features
            test_y = bundle.test_labels

        train_x, train_y = _stratified_data_subset(
            train_x,
            train_y,
            config.data_fraction,
            config.seed + 17,
        )

        train_x, probe_x, train_y, probe_y = train_test_split(
            train_x,
            train_y,
            test_size=config.probe_fraction,
            random_state=config.seed + 1,
            stratify=train_y,
        )
        client_indices = _dirichlet_partition(
            labels=train_y,
            num_clients=config.num_clients,
            alpha=config.dirichlet_alpha,
            min_client_size=config.min_client_size,
            seed=config.seed,
        )
    else:
        if bundle.test_features is None or bundle.test_labels is None:
            raise ValueError(f'{config.dataset_name} requires an explicit test split when predefined client indices are used.')
        selected_x, selected_y, selected_client_indices = _select_predefined_clients(
            bundle.features,
            bundle.labels,
            bundle.client_indices,
            config.num_clients,
            config.seed,
        )
        selected_x, selected_y, selected_client_indices = _subsample_predefined_client_data(
            selected_x,
            selected_y,
            selected_client_indices,
            config.data_fraction,
            config.seed + 17,
        )
        train_x, probe_x, train_y, probe_y, client_indices = _extract_probe_from_predefined_clients(
            selected_x,
            selected_y,
            selected_client_indices,
            config.probe_fraction,
            config.seed + 1,
        )
        test_x = bundle.test_features
        test_y = bundle.test_labels

    feature_shape = tuple(int(dim) for dim in train_x.shape[1:])
    input_dim = int(np.prod(feature_shape))
    num_classes = int(max(train_y.max(), test_y.max())) + 1

    if trigger_kind == 'feature_stamp':
        train_x, probe_x, test_x = _standardize_tabular(train_x, probe_x, test_x)
        trigger_indices = _tabular_trigger_indices(train_x)
        trigger_value = 4.0
    else:
        trigger_indices = ()
        trigger_value = 1.0

    rare_labels = _resolve_rare_labels(train_y, tuple(config.rare_labels))
    if config.malicious_fraction <= 0.0:
        malicious_clients: set[int] = set()
    else:
        malicious_client_count = max(1, int(len(client_indices) * config.malicious_fraction))
        rng = np.random.default_rng(config.seed + 2)
        malicious_clients = set(rng.choice(len(client_indices), size=malicious_client_count, replace=False).tolist())
    client_profiles = _build_client_profiles(
        train_y=train_y,
        client_indices=client_indices,
        num_classes=num_classes,
        rare_labels=rare_labels,
        bandwidth_levels=config.bandwidth_levels,
        malicious_clients=malicious_clients,
    )
    client_profiles = _apply_rarity_information_mode(client_profiles, config.rarity_information_mode)
    backdoor_target = config.backdoor_target if 0 <= config.backdoor_target < num_classes else 0
    model_spec = build_model_spec(
        input_dim=input_dim,
        num_classes=num_classes,
        feature_shape=feature_shape,
        hidden_dims=config.hidden_dims,
        model_family=config.model_family,
        channel_dims=config.channel_dims,
        cnn_hidden_dim=config.cnn_hidden_dim,
        cnn_variant=config.cnn_variant,
    )

    return TaskData(
        dataset_name=config.dataset_name,
        train_dataset=_tensor_dataset(train_x, train_y),
        probe_dataset=_tensor_dataset(probe_x, probe_y),
        test_dataset=_tensor_dataset(test_x, test_y),
        client_indices=client_indices,
        client_profiles=client_profiles,
        input_dim=input_dim,
        feature_shape=feature_shape,
        num_classes=num_classes,
        rare_labels=rare_labels,
        backdoor_target=backdoor_target,
        trigger_kind=trigger_kind,
        trigger_indices=trigger_indices,
        trigger_value=trigger_value,
        model_spec=model_spec,
        image_trigger_variant=config.image_trigger_variant,
    )


def build_digits_task(config: SimulationConfig) -> TaskData:
    return build_task(replace(config, dataset_name='sklearn_digits'))
