"""Load Kronos at the pre-registered code commit and weight revisions, in inference mode (PREREG §1)."""
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "Kronos"
# Same revisions upstream's own regression test pins (vendor/Kronos/tests/test_kronos_regression.py @ 67b630e).
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"      # NeoQuasar/Kronos-small
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"  # NeoQuasar/Kronos-Tokenizer-base


def kronos_classes():
    """Return (Kronos, KronosTokenizer, KronosPredictor) from vendor/Kronos."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from model import Kronos, KronosPredictor, KronosTokenizer
    return Kronos, KronosTokenizer, KronosPredictor


def load_kronos(model_id: str = "NeoQuasar/Kronos-small", tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
                device: str = "cpu", model_revision: str | None = MODEL_REVISION,
                tokenizer_revision: str | None = TOKENIZER_REVISION):
    """Load tokenizer + model with .eval() forced (upstream #415: from_pretrained can leave dropout on)."""
    Kronos, KronosTokenizer, _ = kronos_classes()
    tok = KronosTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision).to(device).eval()
    model = Kronos.from_pretrained(model_id, revision=model_revision).to(device).eval()
    if any(m.training for m in [*tok.modules(), *model.modules()]):
        raise RuntimeError("Kronos is not in eval mode")
    meta = {"model_id": model_id, "model_revision": model_revision,
            "tokenizer_id": tokenizer_id, "tokenizer_revision": tokenizer_revision}
    return tok, model, meta


def make_predictor(tok, model, device: str = "cpu"):
    _, _, KronosPredictor = kronos_classes()
    return KronosPredictor(model, tok, device=device, max_context=512)
