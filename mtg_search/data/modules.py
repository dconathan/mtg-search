from __future__ import annotations
from typing import Optional, List
import logging
import pickle
from tqdm import tqdm
from multiprocessing import cpu_count
from pathlib import Path
from hashlib import md5

from pytorch_lightning import LightningDataModule
from rank_bm25 import BM25Okapi
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mtg_search.data.classes import Card, Cards, Sample, TrainSample, TrainBatch
from mtg_search.constants import DATA_MODULE_PICKLE, PREPROCESSED_DIR

import random


logger = logging.getLogger(__name__)


class IRModule(LightningDataModule):
    def __init__(self, samples: List[Sample], batch_size: int = 8):

        self.samples = samples
        self.batch_size = batch_size
        self.train = []
        self.test = []
        self.val = []
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        # set seed so train/val/test split is consistent
        rng = random.Random(34608)
        for sample in samples:
            r = rng.random()
            if r > 0.15:
                self.train.append(sample)
            elif 0.15 > r > 0.1:
                self.val.append(sample)
            else:
                self.test.append(sample)
        logger.info(
            f"{len(samples)} split into {len(self.train)}/{len(self.val)}/{len(self.test)} train/val/test"
        )
        super().__init__()

    def submodule(self, ratio=10):
        samples = random.sample(self.samples, k=len(self.samples) // ratio)
        return IRModule(samples, batch_size=min(self.batch_size, 2))

    @property
    def corpus(self) -> List[str]:
        return [s.context for s in self.samples]

    @property
    def all_text(self) -> List[str]:
        return [s.query for s in self.samples] + self.corpus

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def dataloder(self, dataset: IRDataset) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=IRDataset.collate,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def train_dataloader(self) -> DataLoader:
        self._train_dataset = self._train_dataset or IRDataset(self.train, self.corpus)
        if not self._train_dataset.preprocessed:
            self._train_dataset.preprocess()
        return self.dataloder(self._train_dataset)

    def test_dataloader(self) -> DataLoader:
        self._test_dataset = self._test_dataset or IRDataset(self.test, self.corpus)
        if not self._test_dataset.preprocessed:
            self._test_dataset.preprocess()
        return self.dataloder(self._test_dataset)

    def val_dataloader(self) -> DataLoader:
        self._val_dataset = self._val_dataset or IRDataset(self.val, self.corpus)
        if not self._val_dataset.preprocessed:
            self._val_dataset.preprocess()
        return self.dataloder(self._val_dataset)

    @classmethod
    def load(cls):
        if DATA_MODULE_PICKLE.exists():
            with DATA_MODULE_PICKLE.open("rb") as f:
                return pickle.load(f)
        cards = Cards.load()
        samples = Sample.from_cards(cards)
        datamodule = cls(samples=samples)
        with DATA_MODULE_PICKLE.open("wb") as f:
            pickle.dump(datamodule, f)
        return datamodule


RNG = random.Random(615)


class IRDataset(Dataset):
    def __init__(self, samples: List[Sample], corpus: List[str], neg_sampling: int = 2):
        """
        TODO docstring
        """
        self.samples = samples
        self._samples = []
        self.corpus = corpus
        self.index = BM25Okapi(corpus)
        if neg_sampling < 2:
            logger.warning(
                "neg_sampling for IRDataset should be at least 2, setting to 2"
            )
            neg_sampling = 2
        self.neg_sampling = neg_sampling

    @property
    def preprocessed_filename(self) -> Path:
        uid = md5(
            (str(self.samples) + str(self.corpus) + str(self.neg_sampling)).encode()
        ).hexdigest()
        return PREPROCESSED_DIR / uid

    @property
    def preprocessed(self):
        return len(self) == len(self._samples)

    def preprocess(self):
        preprocessed_filename = self.preprocessed_filename
        logger.info(f"checking if preprocessed file {preprocessed_filename} exists")
        if preprocessed_filename.exists():
            with preprocessed_filename.open("rb") as f:
                self._samples = pickle.load(f)
            return
        logger.info(
            f"{preprocessed_filename} doesn't exist, preprocessing {len(self)} train samples"
        )
        dataloader = DataLoader(self, num_workers=cpu_count(), batch_size=None)
        for sample in tqdm(iter(dataloader), total=len(self)):
            self._samples.append(sample)
        PREPROCESSED_DIR.mkdir(exist_ok=True, parents=True)
        with self.preprocessed_filename.open("wb") as f:
            pickle.dump(self._samples, f)

    def __getitem__(self, item):

        if self.preprocessed:
            return self._samples[item]

        sample = self.samples[item]

        # we sample adversarial samples (similar contexts) for half our negatives
        n_random = self.neg_sampling // 2
        n_adversarial = self.neg_sampling - n_random
        # + 1 because we need to remove the positive
        ncs = self.index.get_top_n(sample.context, self.corpus, n_adversarial + 1)
        ncs = [nc for nc in ncs if nc != sample.context][: n_adversarial - 1]

        # sample n_random negative contexts
        seen_contexts = set([sample.context] + ncs)
        for _ in range(n_random):
            # this ensures we don't sample the positive or a duplicate negative
            nc = sample.context
            while nc in seen_contexts:
                nc = RNG.choice(self.corpus)
            ncs.append(nc)
            seen_contexts.add(nc)
        return TrainSample(query=sample.query, positive=sample.context, negatives=ncs)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate(samples: List[TrainSample]) -> TrainBatch:
        pcs = set()
        batch = TrainBatch()
        for sample in samples:
            if sample.positive in pcs:
                # this shouldn't be possible but just to be sure
                continue
            batch.queries.append(sample.query)
            batch.contexts.append(sample.positive)
            pcs.add(sample.positive)
        for sample in samples:
            for nc in sample.negatives:
                if nc not in pcs:
                    batch.contexts.append(nc)
        return batch
