import unittest

try:
    import torch
    from src.llm.backend import LocalLLM
    from src.llm.config import LocalModelConfig
except (ImportError, AttributeError):
    torch = None


@unittest.skipIf(torch is None, "PyTorch runtime not installed in lightweight test interpreter")
class LeftPaddedBatchDecodeTests(unittest.TestCase):
    def test_mixed_prompt_lengths_decode_continuation_only(self):
        class Batch(dict):
            def to(self, _device): return self

        class Tokenizer:
            padding_side = "left"
            def apply_chat_template(self, messages, **_kwargs):
                return messages[-1]["content"]
            def __call__(self, texts, **_kwargs):
                rows = [[index + 1 for index, _ in enumerate(text.split())] for text in texts]
                width = max(map(len, rows))
                ids = [[0] * (width - len(row)) + row for row in rows]
                mask = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
                return Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))
            def decode(self, tokens, **_kwargs):
                return ",".join(str(int(value)) for value in tokens)

        class Model:
            device = "cpu"
            def generate(self, input_ids, attention_mask, **_kwargs):
                suffix = torch.tensor([[101, 201], [102, 202]])
                return torch.cat((input_ids, suffix), dim=1)

        config = LocalModelConfig("fake", None, None, False, True, 8)
        llm = LocalLLM(config)
        llm.tokenizer, llm.model = Tokenizer(), Model()
        outputs = llm.generate_batch([("s", "short"), ("s", "a much longer prompt")], batch_size=2)
        self.assertEqual(["101,201", "102,202"], outputs)
        self.assertEqual([1, 4], [row["input_tokens"] for row in llm.generation_stats])


if __name__ == "__main__": unittest.main()
