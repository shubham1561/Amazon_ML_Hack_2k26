"""Learn an Indic-romanisation -> Latin word dictionary from the TRAINING pairs.

Source 2/3 often carry Indian names in native script (गुड टेक -> unidecode 'gudd ttek')
while Source 1 has the Latin spelling ('good tech'). Aligning the tokens of matched
pairs with equal token counts gives a word-level transliteration dictionary. Only
words observed across >= MIN_ENTITIES distinct businesses are kept, so the mapping is
general vocabulary (good, tech, foods, ...), not memorised entity names.

    python src/translit.py        # builds work/translit.pkl from train
"""
import collections
import pickle

import config

MIN_ENTITIES = 3
MIN_SHARE = 0.6


def translit_path():
    return config.work_path("translit.pkl")


def build():
    from data import split_by_country
    from normalize import strip_indic_legal
    from preprocess import load_gt
    gt = load_gt()
    counts = collections.defaultdict(collections.Counter)
    ents = collections.defaultdict(set)
    for c in split_by_country("train"):
        s1 = pickle.load(open(config.work_path("country", f"train_{c}_s1.pkl"), "rb"))
        pool = pickle.load(open(config.work_path("country", f"train_{c}_pool.pkl"), "rb"))
        ind = pool[pool["indic"] == 1]
        if len(ind) == 0:
            continue
        rom = dict(zip(ind["entity_id"].values, ind["core"].values))
        for sid, core in zip(s1["entity_id"].values, s1["core"].values):
            t1 = core.split()
            for m in gt.get(sid, ()):
                r = rom.get(m)
                if r is None:
                    continue
                t2 = strip_indic_legal(r.split())
                if len(t1) != len(t2):
                    continue
                for a, b in zip(t2, t1):
                    if a != b:
                        counts[a][b] += 1
                        ents[(a, b)].add(sid)
    mapping = {}
    for a, cnt in counts.items():
        b, n = cnt.most_common(1)[0]
        if n / sum(cnt.values()) >= MIN_SHARE and len(ents[(a, b)]) >= MIN_ENTITIES:
            mapping[a] = b
    with open(translit_path(), "wb") as f:
        pickle.dump(mapping, f)
    print(f"[translit] {len(mapping):,} words learned from {len(counts):,} romanised forms", flush=True)
    return mapping


def load():
    try:
        with open(translit_path(), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


if __name__ == "__main__":
    m = build()
    for k in list(m)[:30]:
        print(k, "->", m[k])
