from logging import getLogger
from typing import List

logger = getLogger()

class Tokenizer:
    vocab_size: int = 256 + 3
    """tokenizing and encoding/decoding text using UTF8."""
    def __init__(self):
        """
        Initializes the Tokenizer.
        """
        # BOS / EOS token IDs
        self.pad_id: int = 256
        self.bos_id: int = 257
        self.eos_id: int = 258
        logger.info(
            f"#BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def encode(self, text: str, bos: bool, eos: bool, pad: bool = False, max_length: int = -1) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            text (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(text) is str
        tokens = list(text.encode("utf8"))
        numSpecials = sum([bos, eos])

        if pad and max_length > 0:
            tokens = tokens[:max_length - numSpecials]
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens += [self.eos_id]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            tokens (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Remove BOS/EOS/PAD tokens
        return bytes(filter(lambda x: not x in [self.bos_id, self.eos_id, self.pad_id], tokens)).decode("utf8")