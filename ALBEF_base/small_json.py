import json

with open("data/flickr30k_test_filtered.json", "r") as f:
    data = json.load(f)

subset = data[:100]  # first 500 captions/images

with open("data/flickr30k_test_small.json", "w") as f:
    json.dump(subset, f)
