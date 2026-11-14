"""The model list the chat offers: every entry is complete, and vision is flagged on the right ones."""

from app.schemas.chat import AVAILABLE_MODELS, DEFAULT_MODEL

REQUIRED = {"id", "name", "desc", "inputPrice", "outputPrice", "context", "contextWindow", "size", "effortOptions", "huggingFaceId"}


def test_every_model_is_complete_and_ids_are_unique():
    ids = [m["id"] for m in AVAILABLE_MODELS]
    assert len(ids) == len(set(ids))
    for m in AVAILABLE_MODELS:
        assert REQUIRED <= set(m), (m["id"], REQUIRED - set(m))
        assert m["contextWindow"] > 0 and m["effortOptions"] and all(isinstance(o, str) for o in m["effortOptions"])
    assert DEFAULT_MODEL in ids


def test_qwen_3_6_is_offered_with_vision_and_the_qwen_thinking_toggle():
    m = next(m for m in AVAILABLE_MODELS if m["id"] == "qwen/qwen3.6-35b-a3b")
    assert m["vision"] is True and m["effortOptions"] == ["none", "high"]
    assert m["contextWindow"] == 262_000 and m["huggingFaceId"] == "Qwen/Qwen3.6-35B-A3B"


def test_the_models_that_can_see_images_are_exactly_the_flagged_ones():
    from app.agents.vision import model_supports_vision

    for m in AVAILABLE_MODELS:
        assert model_supports_vision(m["id"]) == bool(m.get("vision")), m["id"]
