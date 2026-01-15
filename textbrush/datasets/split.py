"""
Dataset splitter.
"""

import torch.utils.data as torchdata


def split_ordered(
    dataset: torchdata.Dataset,
    ratios: list[float],
) -> list[torchdata.Subset]:
    """
    Split dataset into subsets, keeping the order.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6

    dataset_size = len(dataset)  # type: ignore[arg-type]

    cumsum = [0.0]
    for r in ratios:
        cumsum.append(cumsum[-1] + (r * dataset_size))
    indexes = [round(s) for s in cumsum]
    indexes[-1] = dataset_size

    subsets = [torchdata.Subset(dataset, range(start, end)) for start, end in zip(indexes[:-1], indexes[1:])]

    return subsets


def split_random(
    dataset: torchdata.Dataset,
    ratios: list[float],
) -> list[torchdata.Subset]:
    """
    Split dataset into subsets randomly.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6

    subsets = torchdata.random_split(dataset, ratios)

    return subsets
