# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import typing

from itertools import islice
from torch.nn.utils.rnn import pad_sequence


class MultiLabelEncoder:
    def __init__(self, all_labels: typing.List[str], level: int = 4):
        """
        Map an SF string or list of SF strings
        to a binary vector.
        """
        self.level = level  # level to truncate SF prediction to
        self.labels = list(set(truncate_level(all_labels, self.level)))

    def encode(self, sf_lst) -> typing.List[int]:
        sf_lst = set(truncate_level(sf_lst, self.level))
        onehot = [0] * len(self.labels)
        for idx, label in enumerate(self.labels):
            onehot[idx] = int(label in sf_lst)
        return onehot

    def encode_probs(self, sf_dct: typing.Dict[str, float]) -> typing.List[float]:
        """
        sf_dct is a dictionary of {label: confidence}
        """
        onehot = [0.0] * len(self.labels)
        for label, prob in sf_dct.items():
            onehot[self.labels.index(truncate_level(label, self.level))] = prob
        return onehot


def truncate_level(
    sf_nums: typing.List[str] | str, level: int
) -> typing.List[str] | str:
    if isinstance(sf_nums, list):
        if level == 0:
            return [item.split(".")[0] for item in sf_nums]
        elif level == 1:
            return [".".join(item.split(".")[0:1]) for item in sf_nums]
        elif level == 2:
            return [".".join(item.split(".")[0:2]) for item in sf_nums]
        elif level == 3:
            return [".".join(item.split(".")[0:3]) for item in sf_nums]
        else:
            return sf_nums
    elif isinstance(sf_nums, str):
        if level == 0:
            return sf_nums.split(".")[0]
        elif level == 1:
            return ".".join(sf_nums.split(".")[0:1])
        elif level == 2:
            return ".".join(sf_nums.split(".")[0:2])
        elif level == 3:
            return ".".join(sf_nums.split(".")[0:3])
        else:
            return sf_nums
    else:
        raise ValueError(
            f"Input must be either a string or a list, you have passed: {type(sf_nums)}"
        )


def get_embedding(filepath: str, per_residue: bool = True) -> torch.Tensor:
    """
    Args:
        filepath: path to embedding file.
        per_residue: indicates which type of ESM
            embeddings we wish to load.

    Returns:
        ESM embedding.
    """
    emb = torch.load(filepath)
    if per_residue:
        # The new embedding generation script saves the tensor directly.
        # The previous format was a dictionary. We can just return the tensor.
        return torch.Tensor(emb)
    else:
        # The old format for mean representations was a dictionary.
        # This path may need adjustment if mean embeddings are generated
        # with the new script.
        return torch.Tensor(emb["mean_representations"][33])


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def convert_sf_string_to_list(sf_string: str) -> typing.List[str]:
    """
    Args:
        sf_string: SF number represented as a string.
            e.g. '1.2.3.4'

    Returns:
        Taking the above example,
        '1.2.3.4' goes to
        ['1.', '1.2.', '1.2.3.', '1.2.3.4.']
    """
    parts = sf_string.split(".")
    return [".".join(parts[: i + 1]) + "." for i in range(len(parts))]


def remove_overlap(
    ids_a: typing.List[str], ids_b: typing.List[str]
) -> typing.List[str]:
    """
    Returns ids_a - ids_b
    """
    return list(set(ids_a).difference(ids_b))


def pad_batch_embddings_and_labels(
    batch: typing.List[typing.Tuple], padding_value: int = 0
):
    """
    Return shape:
        [b_size, max_seq_len, embedding_dim]
    """
    all_embs = []
    all_labels = []
    for item in batch:
        all_embs.append(item[0])
        all_labels.append(item[1])  # second element of tuple is the id of the sequence
    padded_embs = pad_sequence(all_embs, batch_first=True, padding_value=padding_value)
    return padded_embs, all_labels
