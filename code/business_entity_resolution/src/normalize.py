"""Text normalization for business names and addresses.

Everything is ASCII-folded (unidecode), lower-cased and canonicalised so that the
noise operations seen in the data (abbreviations, legal-form variants, junk
prefixes, OCR digit swaps, alias markers, reordered components, zero-padded
house numbers, ...) collapse to the same representation on both sides of a pair.
The same code is applied to every source and every country (open set).
"""
import re

from unidecode import unidecode

# ----------------------------------------------------------------------------- names
_RE_INDIC = re.compile(r"[ऀ-෿]")
_RE_ID = re.compile(r"\(?\s*\bid\s*:\s*\d+\s*\)?")
_RE_URL = re.compile(r"https?://|\bwww\.")
_RE_TLD = re.compile(r"\.(?:com|in|net|org|co|fr|biz|info|io|us)\b")
_RE_PHONE = re.compile(r"\d{7,}")
_RE_ALIAS = re.compile(
    r"\s(?:formerly known as|trading as|d\s?/\s?b\s?/\s?a|a\s?/\s?k\s?/\s?a|f\s?/\s?k\s?/\s?a|"
    r"dba|aka|fka|formerly|nee|t\s?/\s?a)\s*:?\s"
)
_RE_NONALNUM = re.compile(r"[^a-z0-9]+")
_OCR = str.maketrans("015837469", "olsbetagg")

LEGAL = {
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt",
    "limited": "ltd", "ltd": "ltd",
    "llc": "llc", "inc": "inc", "incorporated": "inc", "incorp": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc", "pllc": "pllc", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "ei": "ei", "eirl": "eirl", "gie": "gie", "scop": "scop", "scp": "scp",
    "selarl": "selarl", "scm": "scm", "sca": "sca", "selas": "selas",
    "ets": "ets", "etablissements": "ets",
}
HONORIFICS = {"mr", "mrs", "ms", "dr", "smt", "shri", "sri", "sh", "mss"}
NAME_STOP = {"and", "the", "of", "et"}
FILLERS = {"center", "centre", "services", "service", "partners", "group", "groupe",
           "de", "du", "la", "le", "les", "des"}


def _ocr_fix(tok):
    """'5tartups' -> 'startups', 'br0de' -> 'brode', '6lobal' -> 'global', '6ta' -> 'gta'."""
    if tok.isalpha() or tok.isdigit():
        return tok
    nd = sum(ch.isdigit() for ch in tok)
    nl = len(tok) - nd
    if nd <= 2 and (nl >= 3 or (nl >= 2 and tok[0].isdigit() and tok[1:].isalpha())):
        return tok.translate(_OCR)
    return tok


def _name_tokens(s):
    s = s.replace(".", "").replace("'", "")
    toks = _RE_NONALNUM.sub(" ", s).split()
    out = []
    for t in toks:
        t = _ocr_fix(t)
        if t in HONORIFICS or t in NAME_STOP:
            continue
        if out and out[-1] == t:           # "Galaxy Galaxy" -> "galaxy"
            continue
        out.append(t)
    return out


def _split_legal(toks):
    core, legal = [], []
    for t in toks:
        c = LEGAL.get(t)
        if c is None:
            core.append(t)
        else:
            legal.append(c)
    if not core:                            # name made only of legal words: keep them
        core = list(toks)
    return core, legal


def norm_name(raw):
    """Return (core, legal, alias_core, indic_flag)."""
    raw = raw or ""
    indic = 1 if _RE_INDIC.search(raw) else 0
    s = unidecode(raw).lower()
    s = _RE_ID.sub(" ", s)
    s = _RE_URL.sub(" ", s)
    s = _RE_TLD.sub(" ", s)
    s = _RE_PHONE.sub(" ", s)
    s = s.replace("#", " ").replace("&", " and ")
    parts = _RE_ALIAS.split(" " + s + " ")
    alias = ""
    if len(parts) > 1 and parts[0].strip() and parts[-1].strip():
        a_core, _ = _split_legal(_name_tokens(parts[-1]))
        alias = " ".join(a_core)
        s = " ".join(parts)
    core, legal = _split_legal(_name_tokens(s))
    if indic and len(core) > 1:             # romanised legal words: praaivett limittedd, praa li
        kept = [t for t in core if _skel_tok(t) not in _INDIC_LEGAL_SKEL]
        if kept:
            legal = legal + [_INDIC_LEGAL_SKEL[_skel_tok(t)] for t in core if _skel_tok(t) in _INDIC_LEGAL_SKEL]
            core = kept
    if len(core) == 1 and len(core[0]) >= 8 and core[0].endswith("com"):
        core = [core[0][:-3]]               # "fiomolecularcom" -> "fiomolecular"
    return " ".join(core), " ".join(sorted(set(legal))), alias, indic


# ----------------------------------------------------------------------------- addresses
_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
}
_US_STATES_MULTI = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "south carolina": "sc", "south dakota": "sd",
    "rhode island": "ri", "west virginia": "wv", "district of columbia": "dc",
}
_IN_STATES = {
    "maharashtra": "mh", "karnataka": "ka", "gujarat": "gj", "rajasthan": "rj", "punjab": "pb",
    "haryana": "hr", "kerala": "kl", "keralam": "kl", "odisha": "od", "orissa": "od", "bihar": "br",
    "jharkhand": "jh", "chhattisgarh": "cg", "chattisgarh": "cg", "uttarakhand": "uk",
    "uttaranchal": "uk", "telangana": "tg", "ts": "tg", "assam": "as", "goa": "ga", "delhi": "dl",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "sikkim": "sk", "tripura": "tr",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "ladakh": "la",
}
_IN_STATES_MULTI = {
    "uttar pradesh": "up", "madhya pradesh": "mp", "andhra pradesh": "ap", "himachal pradesh": "hp",
    "arunachal pradesh": "ar", "tamil nadu": "tn", "west bengal": "wb", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "andaman and nicobar": "an",
}
_MULTI = {**_US_STATES_MULTI, **_IN_STATES_MULTI}
_RE_MULTI = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _MULTI), key=len, reverse=True)) + r")\b")

ABBR = {
    "road": "rd", "street": "st", "str": "st", "avenue": "ave", "av": "ave", "drive": "dr",
    "drv": "dr", "court": "ct", "crt": "ct", "lane": "ln", "boulevard": "blvd", "bd": "blvd",
    "boul": "blvd", "bvd": "blvd", "place": "pl", "circle": "cir", "parkway": "pkwy",
    "highway": "hwy", "terrace": "ter", "terr": "ter", "trail": "trl", "square": "sq",
    "saint": "st", "sainte": "ste", "r": "rue", "all": "allee", "impasse": "imp", "route": "rte",
    "chemin": "ch", "chem": "ch", "faubourg": "fbg", "cours": "crs", "north": "n", "south": "s",
    "east": "e", "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se",
    "southwest": "sw", "fort": "ft", "mount": "mt", "center": "ctr", "centre": "ctr",
    "heights": "hts", "floor": "fl", "building": "bldg", "number": "no", "nagar": "ngr",
    "sector": "sec", "opposite": "opp", "near": "nr", "expressway": "expy", "junction": "jct",
    "plaza": "plz", "point": "pt", "apartment": "apt", "apartments": "apt", "apts": "apt",
    "society": "soc", "colony": "col", "bombay": "mumbai", "bangalore": "bengaluru",
    "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram", "poona": "pune",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12",
}
ABBR.update(_US_STATES)
ABBR.update(_IN_STATES)
ADDR_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "of", "the", "and", "null", "none",
             "nan", "no", "nr", "opp", "behind", "beside", "cdp"}

_RE_POBOX = re.compile(r"\bp\s*o\s*box\s*#?\s*(\d+)")
_RE_UNIT = re.compile(
    r"\b(?:unit|suite|ste|apt|apartment)\b(?:\s*(?:unit|suite|ste|apt|apartment|#))*\s*#?\s*([a-z0-9-]+)"
)
_RE_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_RE_DIGITS = re.compile(r"\d+")


def norm_addr(raw):
    """Return (addr_norm, nums, unit, pobox).

    addr_norm: canonical token string (numbers included, leading zeros stripped)
    nums: space-separated distinct numbers in order of appearance (units / PO boxes excluded)
    """
    raw = raw or ""
    if not raw.strip():
        return "", "", "", ""
    s = unidecode(raw).lower().replace(".", " ").replace("'", "")
    pob = _RE_POBOX.findall(s)
    s = _RE_POBOX.sub(" ", s)
    units = _RE_UNIT.findall(s)
    s = _RE_UNIT.sub(" ", s)
    s = _RE_ORD.sub(r"\1", s)
    s = _RE_NONALNUM.sub(" ", s)
    s = _RE_MULTI.sub(lambda m: _MULTI[m.group(1)], s)
    toks, nums, seen = [], [], set()
    for t in s.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
            if t not in seen:
                seen.add(t)
                nums.append(t)
            toks.append(t)
            continue
        if t[0].isdigit():                  # 7a, 9809a, 16410a -> number + suffix
            m = _RE_DIGITS.match(t)
            n = m.group(0).lstrip("0") or "0"
            if n not in seen:
                seen.add(n)
                nums.append(n)
            toks.append(n)
            continue
        t = ABBR.get(t, t)
        if t in ADDR_STOP:
            continue
        if t.isdigit() and t not in seen:   # "sixth" -> "6"
            seen.add(t)
            nums.append(t)
        toks.append(t)
    unit = ""
    if units:
        u = units[0]
        m = _RE_DIGITS.search(u)
        unit = (m.group(0).lstrip("0") or "0") if m else u
    pobox = pob[0].lstrip("0") if pob else ""
    return " ".join(toks), " ".join(nums), unit, pobox


# ----------------------------------------------------------------------------- phonetic skeleton
_SKEL_SUBS = (("ph", "f"), ("ch", "k"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"),
              ("gh", "g"), ("kh", "k"), ("q", "k"), ("z", "s"), ("x", "ks"), ("b", "v"), ("w", "v"))
_RE_SOFT_C = re.compile(r"c(?=[eiy])")
_RE_SOFT_G = re.compile(r"g(?=[ei])")
_RE_VOWELS = re.compile(r"[aeiouy]")
_RE_NASAL = re.compile(r"n(?=[bcdfgjklmpqrstvxz])")
_RE_REPEAT = re.compile(r"(.)\1+")


def _skel_tok(t):
    """Consonant skeleton of one token; the same rules are applied to Latin and romanised
    Indic spellings so that 'digital'/'ddijittl', 'city'/'sittii', 'finance'/'phaainens' agree."""
    if t.isdigit():
        return t
    t = _RE_SOFT_C.sub("s", t)
    t = _RE_SOFT_G.sub("j", t)
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    t = t.replace("c", "k")
    head = "a" if t[0] in "aeiouy" else t[0]
    rest = _RE_NASAL.sub("", _RE_VOWELS.sub("", t[1:]))
    return _RE_REPEAT.sub(r"\1", head + rest)


def skeleton(text):
    """Consonant skeleton, robust to romanisation differences (for Indic-script names)."""
    return " ".join(_skel_tok(t) for t in text.split())


# skeletons of romanised Indic legal words (प्राइवेट लिमिटेड -> praaivett limittedd -> prvt lmtd)
_INDIC_LEGAL_SKEL = {"prvt": "pvt", "prvr": "pvt", "pr": "pvt", "prv": "pvt", "lmtd": "ltd", "lmrd": "ltd",
                     "l": "ltd", "lmt": "ltd", "kmpn": "co", "elp": "llp", "elelp": "llp", "alp": "llp",
                     "lp": "llp"}


def strip_indic_legal(tokens):
    kept = [t for t in tokens if _skel_tok(t) not in _INDIC_LEGAL_SKEL]
    return kept if kept else tokens


def normalize_records(names, addrs):
    """Vector version used by the multiprocessing workers."""
    res = []
    for n, a in zip(names, addrs):
        core, legal, alias, indic = norm_name(n)
        an, nums, unit, pobox = norm_addr(a)
        res.append((core, legal, alias, indic, an, nums, unit, pobox))
    return res
