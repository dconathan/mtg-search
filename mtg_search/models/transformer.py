import comet_ml
from dataclasses import dataclass
import os
from typing import List
import logging
import argparse

import torch
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers.optimization import AdamW
from transformers.tokenization_utils import BatchEncoding
from transformers.models.bert import BertModel, BertTokenizerFast
from transformers.models.roberta import (
    RobertaModel,
    RobertaTokenizerFast,
    RobertaConfig,
)
from tokenizers.implementations import ByteLevelBPETokenizer

from mtg_search.data.classes import TrainBatch
from mtg_search.data.modules import IRModule
from mtg_search.models.loss import BiEncoderNllLoss
from mtg_search.constants import TOKENIZER_JSON, MODELS_DIR, MODEL_CHECKPOINT_PATH


loss_fn = BiEncoderNllLoss()

logger = logging.getLogger(__name__)


class BaseConfig:
    num_hidden_layers = 2
    num_attention_heads = 2
    hidden_size = 128
    intermediate_size = 128
    max_position_embeddings = 128


def train_tokenizer(datamodule: IRModule):
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    tokenizer.train_from_iterator(
        datamodule.all_text,
        vocab_size=5000,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    )
    TOKENIZER_JSON.parent.mkdir(exist_ok=True)
    tokenizer.save(str(TOKENIZER_JSON))


@dataclass
class Output:
    q: torch.Tensor
    c: torch.Tensor

    def __len__(self):
        return self.q.shape[0]

    def loss(self):
        return loss_fn.calc(self.q, self.c, list(range(len(self))))[0]

    def acc(self):
        preds = torch.matmul(self.q, torch.transpose(self.c, 0, 1)).argmax(dim=1)
        preds = preds.detach().to("cpu")
        return (preds == torch.arange(len(self))).to(torch.float).mean()


@dataclass
class Input:
    q: BatchEncoding
    c: BatchEncoding

    @classmethod
    def from_batch(cls, batch: TrainBatch, tokenizer: BertTokenizerFast):
        q = tokenizer.batch_encode_plus(
            batch.queries,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=BaseConfig.max_position_embeddings - 1,
        )
        c = tokenizer.batch_encode_plus(
            batch.contexts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=BaseConfig.max_position_embeddings - 1,
        )
        return cls(q, c)

    def to(self, *args, **kwargs):
        # for device, etc.
        self.q.to(*args, **kwargs)
        self.c.to(*args, **kwargs)
        return self


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_JSON.parent)

        config = RobertaConfig(
            num_hidden_layers=BaseConfig.num_hidden_layers,
            num_attention_heads=BaseConfig.num_attention_heads,
            hidden_size=BaseConfig.hidden_size,
            intermediate_size=BaseConfig.intermediate_size,
        )

        self.q_encoder = RobertaModel(config)
        self.c_encoder = RobertaModel(config)

    def create_index(self, contexts: List[str], batch_size=32) -> torch.Tensor:
        logger.info(f"creating index on {len(contexts)} contexts")
        cs = []
        with torch.no_grad():
            for c in tqdm(contexts):
                c = self.tokenizer.encode(
                    c,
                    return_tensors="pt",
                    truncation=True,
                    max_length=BaseConfig.max_position_embeddings - 1,
                )
                c = self.c_encoder(c).last_hidden_state[:, 0, :]
                cs.append(c)
        return torch.cat(cs)

    def embed_query(self, q: str) -> torch.Tensor:
        with torch.no_grad():
            q = self.tokenizer.encode(
                q,
                return_tensors="pt",
                truncation=True,
                max_length=BaseConfig.max_position_embeddings - 1,
            )
            q = self.q_encoder(q).last_hidden_state[:, 0, :]
        return q[0]

    def forward(self, batch: Input) -> Output:

        batch.to(self.device)

        q = self.q_encoder(**batch.q).last_hidden_state[:, 0, :]
        c = self.c_encoder(**batch.c).last_hidden_state[:, 0, :]

        return Output(q=q, c=c)

    def step(self, batch: TrainBatch) -> Output:
        encoded = Input.from_batch(batch, self.tokenizer)
        output = self(encoded)
        return output

    def training_step(self, batch: TrainBatch, *args, **kwargs):
        output = self.step(batch)
        loss = output.loss()
        self.log_dict({"train_loss": loss, "train_acc": output.acc()})
        return loss

    def validation_step(self, batch: TrainBatch, *args, **kwargs):
        with torch.no_grad():
            output = self.step(batch)
        self.log_dict({"val_loss": output.loss(), "val_acc": output.acc()})

    def test_step(self, batch: TrainBatch, *args, **kwargs):
        with torch.no_grad():
            output = self.step(batch)
        self.log_dict({"test_loss": output.loss(), "test_acc": output.acc()})

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)


def main():

    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-tokenizer", action="store_true")

    args = parser.parse_args()

    datamodule = IRModule.load()
    datamodule.batch_size = args.batch_size

    if args.train_tokenizer:
        train_tokenizer(datamodule)
    # datamodule = datamodule.submodule(ratio=100)

    model = Model()

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        experiment_name="mtg-search",
        log_graph=False,
        log_code=False,
        log_env_details=False,
    )

    key = comet_logger.experiment.get_key()

    callbacks = [
        ModelCheckpoint(
            dirpath=MODELS_DIR, save_top_k=1, monitor="val_acc", filename=key,
        )
    ]

    trainer = Trainer(
        logger=comet_logger,
        max_epochs=100,
        callbacks=callbacks,
        val_check_interval=250,
        num_sanity_val_steps=0,
        gpus=args.gpus,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
