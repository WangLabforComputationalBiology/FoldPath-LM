"""从 vocab.txt 生成 tokenizer.json（简单的 char-level 映射）"""
import json, os

model_path = 'pretrained/progen2-small'
vocab = {}
with open(os.path.join(model_path, 'vocab.txt')) as f:
    for i, line in enumerate(f):
        vocab[line.strip()] = i

# Build tokenizer.json manually
tok = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 0, "content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 1, "content": "<cls>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 2, "content": "<eos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 3, "content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 4, "content": "<mask>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 5, "content": "<null>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
    ],
    "normalizer": {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": False, "strip_accents": False, "lowercase": False},
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
    "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<cls>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 1}}, {"SpecialToken": {"id": "<eos>", "type_id": 2}}], "pair": None},
    "decoder": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "fuse_unk": False,
        "vocab": vocab,
        "merges": []
    }
}

with open(os.path.join(model_path, 'tokenizer.json'), 'w') as f:
    json.dump(tok, f)
print(f'Saved tokenizer.json with {len(vocab)} tokens')
