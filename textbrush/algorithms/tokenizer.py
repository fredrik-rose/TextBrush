"""
Simple character level tokenizer.
"""


class Tokenizer:
    """
    Character level tokenizer.
    """

    def __init__(self, vocab: list[str]):
        assert len(set(vocab)) == len(vocab), "Vocabulary must be unique."

        self.vocab_size = len(vocab)

        self._int_to_token = dict(enumerate(sorted(vocab)))
        self._token_to_int = {token: i for i, token in self._int_to_token.items()}

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        Encode a text string to a list of numbers.
        """
        return [self._token_to_int[c] for c in text]

    def decode(
        self,
        numbers: list[int],
    ) -> str:
        """
        Decode a list of numbers to a text string.
        """
        return "".join(self._int_to_token[i] for i in numbers)
