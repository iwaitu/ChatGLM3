from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/text-embedding-ada-002")
assert tokenizer.encode("hello world") == [15339, 1917]
