from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = self._select(avai_pids)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def _select(self, avai_pids):
        return random.sample(avai_pids, self.num_pids_per_batch)

    def __len__(self):
        return self.length


class PartHardPKSampler(RandomIdentitySampler):
    """PK sampler whose batches hold confusable identities for one body-part slot.

    Per batch: pick a slot, an anchor identity eligible for that slot, then `hard_frac` of the other P-1 identities
    from the anchor's neighbours in that slot (`nbr` [S, C, k], identity labels), the rest at random. With no
    neighbour table set (set_neighbours(None)) it is exactly RandomIdentitySampler. Identity labels must be the
    contiguous train labels the table is indexed by.
    """

    def __init__(self, data_source, batch_size, num_instances, hard_frac=0.5):
        super().__init__(data_source, batch_size, num_instances)
        self.hard_frac = hard_frac
        self.nbr, self.eligible = None, None

    def set_neighbours(self, nbr, eligible=None):
        """nbr: [S, C, k] int array or None; eligible: [S, C] bool array (identities usable as anchors per slot)."""
        self.nbr, self.eligible = nbr, eligible

    def _select(self, avai_pids):
        if self.nbr is None:
            return super()._select(avai_pids)
        P, avai = self.num_pids_per_batch, set(avai_pids)
        slot = random.randrange(self.nbr.shape[0])
        pool = [p for p in avai_pids if self.eligible is None or self.eligible[slot, p]]
        anchor = random.choice(pool or avai_pids)
        cand = [int(p) for p in self.nbr[slot, anchor] if int(p) in avai and int(p) != anchor]
        chosen = [anchor] + random.sample(cand, min(int(round(self.hard_frac * (P - 1))), len(cand)))
        rest = [p for p in avai_pids if p not in set(chosen)]
        return chosen + random.sample(rest, P - len(chosen))

