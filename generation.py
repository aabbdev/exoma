from model import Transformer, ModelArgs
from tokenizer import Tokenizer
import torch

tokenizer = Tokenizer()
model = Transformer(ModelArgs(vocab_size=tokenizer.vocab_size)).eval()

input_ids = torch.tensor(tokenizer.encode("Hello, how are you?", bos=True, eos=True))

output = model(input_ids.unsqueeze(0), start_pos=0)
