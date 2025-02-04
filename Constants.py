PAD = 0
Ks = [5, 10, 20]
print("k:", Ks)

DATASET = "Zvuk"
ENCODER = "THP"  # NHP THP Transformer
ABLATIONs = {
    "w/oWeSch",
    "w/oPopDe",
    "w/oSA",
    "w/oNorm",
    "w/oUSpec",
    "w/oHgcn",
    "w/oDisen",
}
ABLATION = "Full"

user_dict = {
    "ml-1M": 6041,
    "Beauty": 22363,
    "Yelp2023": 48993,
    "Food.com": 7452,
    "Amazon-book": 19804,
    "Kuairand-1k": 1000,  # 16971,
    "Zvuk": 1000,
}

item_dict = {
    "ml-1M": 3955,
    "Beauty": 12101,
    "Yelp2023": 34298,
    "Food.com": 12911,
    "Amazon-book": 22086,
    "Kuairand-1k": 18561,  # 3274710,
    "Zvuk": 19144,
}

ITEM_NUMBER = item_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)

print("Dataset:", DATASET, "#User:", USER_NUMBER, "#Item", ITEM_NUMBER)
print("Encoder: ", ENCODER)
print("ABLATION: ", ABLATION)

DICT = {
    "ml-1M": 2,
    "Beauty": USER_NUMBER,
    "Yelp2023": USER_NUMBER,
    "Food.com": 1e5,
    "Amazon-book": 1e5,
    "Kuairand-1k": USER_NUMBER,
}

NUM_LAYERS_DICT = {
    "ml-1M": 2,
    "Beauty": 2,
    "Yelp2023": 2,
    "Food.com": 1,
    "Amazon-book": 1,
    "Kuairand-1k": 2,
    "Zvuk": 2,
}

NUM_LAYERS = NUM_LAYERS_DICT[DATASET]
