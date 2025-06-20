# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import os
import typing

from collections import defaultdict
from torch.utils.data import Dataset

from utils.dataset_utils import get_embedding, convert_sf_string_to_list


class EmbeddingsAndLabelsDataset(Dataset):
    def __init__(
        self, filepath: str, ids: typing.List[str], id_to_sf, train_on_classes=False
    ) -> None:
        """
        Each label is converted to one set.
        An SF label is represented hierarchically
        as '1.2.3.4' -> ['1.', '1.2.', '1.2.3.', '1.2.3.4']
        """
        self.filepath = filepath
        self.ids = ids
        self.id_to_sf = id_to_sf
        self.train_on_classes = train_on_classes
        if self.ids is None:
            self.embedding_filenames = [f for f in os.listdir(self.filepath)]
        else:
            all_fnames = [f[:-3] for f in os.listdir(self.filepath)]
            id_embs = set(all_fnames).intersection(set(self.ids))
            assert len(id_embs) == len(
                self.ids
            ), f"We only have embeddings for \
                {len(id_embs)} of these ids!\n,  \
                We do not have: {set(self.ids).difference(set(id_embs))}"
            self.embedding_filenames = [id_emb + ".pt" for id_emb in id_embs]
        # if we want to pass over the classes
        if self.train_on_classes:
            sf_to_id = defaultdict(list)
            if self.ids is None:
                for id, sf_nums in self.id_to_sf.items():
                    for sf in sf_nums:
                        sf_to_id[sf].append(id)
            else:  # otherwise use only the ids we need
                for id in id_embs:
                    sf_nums = self.id_to_sf[id]
                    for sf in sf_nums:
                        sf_to_id[sf].append(id)
            self.sf_to_id = sf_to_id
            self.sf_nums = list(sf_to_id.keys())

    def __getitem__(self, idx):
        if self.train_on_classes:
            sf_num = self.sf_nums[idx]
            id = np.random.choice(self.sf_to_id[sf_num])
            emb = get_embedding(self.filepath + "/" + id + ".pt")
            labels = self.id_to_sf[id]
        else:
            emb = get_embedding(self.filepath + "/" + self.embedding_filenames[idx])
            labels = self.id_to_sf[self.embedding_filenames[idx].strip(".pt")]
        labels_as_list = []
        for sf in labels:
            labels_as_list.extend(convert_sf_string_to_list(sf))
        return emb, labels_as_list

    def __len__(self):
        if self.train_on_classes:
            return len(self.sf_nums)
        else:
            return len(self.embedding_filenames)
