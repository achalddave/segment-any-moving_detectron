from .json_dataset import JsonDataset


def load_dataset(name, offsets):
    # Can be overridden for other datasets.
    return JsonDataset(name, offsets)
