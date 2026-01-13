"""
Shakespeare text generator.
"""

from textbrush.datasets import tinyshakespeare
from textbrush.models import gpt

MAX_TOKENS = 8
NUM_LAYERS = 3
NUM_HEADS = 2
EMBEDDED_DIMENSION = 32
FEED_FORWARD_DIMENSION = EMBEDDED_DIMENSION * 4
DROPOUT = 0.2
ATTENTION_DROPOUT = DROPOUT


class Textgenerator:
    """
    Text generator using a GPT model as backend.
    """

    def __init__(self):
        self.dataset = tinyshakespeare.TinyShakespeare(
            block_size=MAX_TOKENS,
        )
        self.model = gpt.GPT(
            vocab_size=self.dataset.vocab_size,
            num_tokens=MAX_TOKENS,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            embed_dim=EMBEDDED_DIMENSION,
            feed_forward_dim=FEED_FORWARD_DIMENSION,
            dropout=DROPOUT,
            attention_dropout=ATTENTION_DROPOUT,
        )

    def __call__(
        self,
        prompt: str,
        length: int,
    ) -> str:
        """
        Generate text given a prompt.
        """
        tokens = self.dataset.encode(prompt)
        generator = self.model.generate(tokens)
        text = self.dataset.decode(next(generator) for _ in range(length))
        return text
