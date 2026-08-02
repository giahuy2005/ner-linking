from .repair_gate import filter_entities
from .sectioner import split_sections_by_header
from .postprocessor import clean_text_for_inference


def __getattr__(name):
    if name in {"NerAssertionModel", "NerEngine"}:
        from .engine import NerAssertionModel, NerEngine

        return {"NerAssertionModel": NerAssertionModel, "NerEngine": NerEngine}[name]
    raise AttributeError(name)

__all__ = [
    "NerAssertionModel",
    "NerEngine",
    "filter_entities",
    "split_sections_by_header",
    "clean_text_for_inference",
]
