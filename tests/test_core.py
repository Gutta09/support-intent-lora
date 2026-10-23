import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from src.app import route
from src.quantize import quantize_dynamic, size_mb
from src.train import trainable_share


def tiny_model(num_labels=5):
    cfg = DistilBertConfig(vocab_size=100, dim=32, n_layers=1, n_heads=2, hidden_dim=64,
                           num_labels=num_labels)
    return DistilBertForSequenceClassification(cfg).eval()


def test_route_escalates_low_confidence():
    probs = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.55, 0.45]])
    out = route(probs, ["card_lost", "refund"], threshold=0.6)
    assert [r["intent"] for r in out] == ["card_lost", "refund", "card_lost"]
    assert [r["escalate"] for r in out] == [False, False, True]


def test_route_threshold_is_strict():
    out = route(torch.tensor([[0.5, 0.5]]), ["a", "b"], threshold=0.5)
    assert out[0]["escalate"] is False


def test_quantization_shrinks_model_and_keeps_shape():
    fp32 = tiny_model()
    int8 = quantize_dynamic(fp32)
    assert size_mb(int8) < size_mb(fp32)
    ids = torch.randint(0, 100, (3, 8))
    with torch.inference_mode():
        a, b = fp32(input_ids=ids).logits, int8(input_ids=ids).logits
    assert a.shape == b.shape == (3, 5)
    assert torch.equal(a.argmax(-1), b.argmax(-1)) or (a - b).abs().max() < 0.5


def test_quantized_linears_are_replaced():
    int8 = quantize_dynamic(tiny_model())
    kinds = {type(m).__name__ for m in int8.modules()}
    assert "Linear" in kinds  # quantized dynamic Linear keeps the class name
    assert any("quantized" in type(m).__module__ for m in int8.modules())


def test_lora_trains_only_a_small_share():
    from peft import LoraConfig, TaskType, get_peft_model
    m = get_peft_model(tiny_model(), LoraConfig(task_type=TaskType.SEQ_CLS, r=4,
                                                target_modules=["q_lin", "v_lin"]))
    trainable, total = trainable_share(m)
    assert 0 < trainable < total * 0.5
