from dataclasses import dataclass, asdict
import os
from typing import List
import logging
import argparse
from pathlib import Path

try:
    import comet_ml
except:
    pass
import torch
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers.optimization import AdamW
from transformers.tokenization_utils import BatchEncoding
from transformers.models.bert import BertModel, BertTokenizerFast

from mtg_search.data.classes import TrainBatch
from mtg_search.data.modules import IRModule
from mtg_search.models.loss import BiEncoderNllLoss
from mtg_search.constants import MODELS_DIR


loss_fn = BiEncoderNllLoss()

logger = logging.getLogger(__name__)


@dataclass
class TinyBertConfig:
    name: str = "prajjwal1/bert-tiny"
    key: str = None


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
        )
        c = tokenizer.batch_encode_plus(
            batch.contexts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return cls(q, c)

    def to(self, *args, **kwargs):
        # for device, etc.
        self.q.to(*args, **kwargs)
        self.c.to(*args, **kwargs)
        return self


class Model(LightningModule):
    def __init__(self, config, tokenizer, q_encoder, c_encoder):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.q_encoder = q_encoder
        self.c_encoder = c_encoder

    @classmethod
    def from_tinybert(cls):
        config = TinyBertConfig()
        tokenizer = BertTokenizerFast.from_pretrained(config.name)
        q_encoder = BertModel.from_pretrained(config.name)
        c_encoder = BertModel.from_pretrained(config.name)
        return cls(config, tokenizer, q_encoder, c_encoder)

    def on_save_checkpoint(self, *args, **kwargs) -> None:
        if not self.config.key:
            return
        checkpoint_dir = MODELS_DIR / self.config.key
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.config, checkpoint_dir / "config.torch")
        torch.save(self.tokenizer, checkpoint_dir / "tokenizer.torch")
        torch.save(self.q_encoder, checkpoint_dir / "q_encoder.torch")
        torch.save(self.c_encoder, checkpoint_dir / "c_encoder.torch")

    @classmethod
    def from_checkpoint_dir(cls, path, q_only=False, c_only=False):
        path = Path(path)
        config = torch.load(path / "config.torch")
        tokenizer = torch.load(path / "tokenizer.torch")
        if c_only:
            q_encoder = None
        else:
            q_encoder = torch.load(path / "q_encoder.torch")
        if q_only:
            c_encoder = None
        else:
            c_encoder = torch.load(path / "c_encoder.torch")
        return cls(config, tokenizer, q_encoder, c_encoder)

    @staticmethod
    def pool(output):
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0, :]
        else:
            raise AttributeError(
                f"don't know how to pool output of type {type(output)}"
            )

    def create_index(self, contexts: List[str]) -> torch.Tensor:
        logger.info(f"creating index on {len(contexts)} contexts")
        cs = []
        with torch.no_grad():
            for c in tqdm(contexts):
                c = self.tokenizer.encode(
                    c,
                    return_tensors="pt",
                    truncation=True,
                )
                c = self.pool(self.c_encoder(c))
                cs.append(c)
        return torch.cat(cs)

    def embed_query(self, q: str) -> torch.Tensor:
        with torch.no_grad():
            q = self.tokenizer.encode(
                q,
                return_tensors="pt",
                truncation=True,
            )
            q = self.pool(self.q_encoder(q))
        return q[0]

    def forward(self, batch: Input) -> Output:

        batch.to(self.device)

        q = self.pool(self.q_encoder(**batch.q))
        c = self.pool(self.c_encoder(**batch.c))

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

    args = parser.parse_args()

    datamodule = IRModule.load()
    datamodule.batch_size = args.batch_size

    # datamodule = datamodule.submodule(ratio=100)

    model = Model.from_tinybert()

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        experiment_name="mtg-search",
        log_graph=False,
        log_code=False,
        log_env_details=False,
        disabled=True,
    )

    key = comet_logger.experiment.get_key()
    model.config.key = key

    callbacks = [
        ModelCheckpoint(
            dirpath=MODELS_DIR,
            save_top_k=1,
            monitor="val_acc",
            filename=key,
        )
    ]

    comet_logger.log_hyperparams(asdict(model.config))

    trainer = Trainer.from_argparse_args(
        args,
        logger=comet_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        val_check_interval=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
