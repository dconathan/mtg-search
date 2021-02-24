from mtg_search.api import main
from mtg_search.models.transformer import Model
from mtg_search.data.classes import Index, Cards, Card
from transformers import BertModel


def test_load_q_encoder():
    model = main.load_q_encoder()
    assert isinstance(model, Model)
    assert isinstance(model.q_encoder, BertModel)


def test_load_c_encoder():
    model = main.load_c_encoder()
    assert isinstance(model, Model)
    assert isinstance(model.c_encoder, BertModel)


def test_load_index():
    index = main.load_index()
    assert isinstance(index, Index)
    assert index.vectors.shape[0] == len(index.cards)


def test_search():
    result = main.search("hello world", top=10)
    assert isinstance(result, list)
    assert len(result) == 10
    for card in result:
        assert isinstance(card, Card)
