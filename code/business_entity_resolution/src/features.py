"""Pair features. Everything is vectorised: rapidfuzz.process.cpdist (multi-threaded C++)
for string similarities, sparse row products for IDF / TF-IDF evidence, numpy for numbers.

Features are country-agnostic (similarities, ratios, frequencies): nothing identifies a
country, so the model transfers to countries unseen in training (France).
"""
import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

from blocking import to_csr, csr_rows

FILLERS = {"center", "centre", "services", "service", "partners", "group", "groupe",
           "de", "du", "la", "le", "les", "des"}


def _cp(a, b, scorer, dtype=np.uint8):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=dtype)


class CountryContext:
    """Per-country record-level statistics shared by all pair-feature calls."""

    def __init__(self, s1, s1_nums, pool, pool_nums, with_tfidf=False):
        self.s1, self.pool = s1, pool
        self.s1_nums, self.pool_nums = s1_nums, pool_nums
        n_all = len(s1) + len(pool)

        # name token IDF (S1 + pool) ----------------------------------------------------------
        vocab = {}
        s1_ip, s1_ids = to_csr(s1["core"].values, vocab)
        p_ip, p_ids = to_csr(pool["core"].values, vocab)
        df = np.bincount(s1_ids, minlength=len(vocab)) + np.bincount(p_ids, minlength=len(vocab))
        self.idf = np.log((n_all + 1) / (df + 1)).astype(np.float32) + 1.0
        self.A_name = sp.csr_matrix((self.idf[s1_ids], s1_ids, s1_ip), shape=(len(s1), len(vocab)))
        self.B_name = sp.csr_matrix((np.ones(len(p_ids), np.float32), p_ids, p_ip), shape=(len(pool), len(vocab)))
        self.B_name_idf = sp.csr_matrix((self.idf[p_ids], p_ids, p_ip), shape=(len(pool), len(vocab)))
        self.s1_name_idf_sum = np.asarray(self.A_name.sum(1)).ravel()
        self.p_name_idf_sum = np.asarray(self.B_name_idf.sum(1)).ravel()

        # address word IDF ---------------------------------------------------------------------
        avocab = {}
        s1_ap, s1_aids = to_csr(s1["addr"].values, avocab)
        p_ap, p_aids = to_csr(pool["addr"].values, avocab)
        adf = np.bincount(s1_aids, minlength=len(avocab)) + np.bincount(p_aids, minlength=len(avocab))
        aidf = np.log((n_all + 1) / (adf + 1)).astype(np.float32) + 1.0
        self.A_addr = sp.csr_matrix((aidf[s1_aids], s1_aids, s1_ap), shape=(len(s1), len(avocab)))
        self.B_addr = sp.csr_matrix((np.ones(len(p_aids), np.float32), p_aids, p_ap), shape=(len(pool), len(avocab)))
        self.B_addr_idf = sp.csr_matrix((aidf[p_aids], p_aids, p_ap), shape=(len(pool), len(avocab)))
        self.s1_addr_idf_sum = np.asarray(self.A_addr.sum(1)).ravel()
        self.p_addr_idf_sum = np.asarray(self.B_addr_idf.sum(1)).ravel()

        # name frequency (ambiguity) -------------------------------------------------------------
        s1_counts = s1["sorted_core"].map(s1["sorted_core"].value_counts())
        self.s1_name_freq = np.log1p(s1_counts.values.astype(np.float32))
        p_counts = pool["sorted_core"].map(pool["sorted_core"].value_counts())
        self.p_name_freq = np.log1p(p_counts.values.astype(np.float32))
        # how many S1 records share the pool record's name
        s1_vc = s1["sorted_core"].value_counts()
        self.p_name_in_s1 = np.log1p(pool["sorted_core"].map(s1_vc).fillna(0).values.astype(np.float32))

        self.s1_core2 = [" ".join(t for t in c.split() if t not in FILLERS) for c in s1["core"].values]
        self.p_core2 = [" ".join(t for t in c.split() if t not in FILLERS) for c in pool["core"].values]

        self.tfidf = None
        if with_tfidf:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                                  sublinear_tf=True, dtype=np.float32)
            vec.fit(np.concatenate([s1["core"].values, pool["core"].values]))
            self.T1 = vec.transform(s1["core"].values)
            self.TP = vec.transform(pool["core"].values)
            self.tfidf = vec


def _rowdot(A, B, i1, ip):
    return np.asarray(A[i1].multiply(B[ip]).sum(1)).ravel()


def _rowmax(A, B, i1, ip):
    return A[i1].multiply(B[ip]).max(axis=1).toarray().ravel()


def _num_feats(n1, n2):
    valid1 = n1 >= 0
    eq = (n1[:, :, None] == n2[:, None, :]) & valid1[:, :, None]
    common = eq.any(axis=2).sum(axis=1).astype(np.uint8)
    has1 = valid1[:, 0]
    has2 = n2[:, 0] >= 0
    first_eq = np.where(has1 & has2, (n1[:, 0] == n2[:, 0]).astype(np.int8), -1).astype(np.int8)
    n_1 = valid1.sum(1)
    n_2 = (n2 >= 0).sum(1)
    denom = np.maximum(n_1 + n_2 - common, 1)
    jac = np.where((n_1 > 0) & (n_2 > 0), common / denom, -1).astype(np.float32)
    return common, first_eq, jac, n_1.astype(np.uint8), n_2.astype(np.uint8)


CHEAP = ["n_tset", "n_tsort", "n_ns_ratio", "n_skel_tset", "n_alias_tset", "a_tset",
         "num_common", "num_first_eq", "p_addr_empty", "p_indic", "keymask", "src",
         "s1_name_freq", "p_name_freq", "n_idf_cov1", "n_idf_cov2", "n_max_idf"]


def cheap_features(ctx, i1, ip, keymask):
    s1, pool = ctx.s1, ctx.pool
    c1, cp = s1["core"].values[i1].tolist(), pool["core"].values[ip].tolist()
    f = {}
    f["n_tset"] = _cp(c1, cp, fuzz.token_set_ratio)
    f["n_tsort"] = _cp(c1, cp, fuzz.token_sort_ratio)
    f["n_ns_ratio"] = _cp(s1["nospace"].values[i1].tolist(), pool["nospace"].values[ip].tolist(), fuzz.ratio)
    f["n_skel_tset"] = _cp(s1["skel"].values[i1].tolist(), pool["skel"].values[ip].tolist(), fuzz.token_set_ratio)
    pa = pool["alias"].values[ip]
    has_alias = pa != ""
    al = np.zeros(len(i1), np.uint8)
    if has_alias.any():
        al[has_alias] = _cp(np.asarray(c1, dtype=object)[has_alias].tolist(), pa[has_alias].tolist(),
                            fuzz.token_set_ratio)
    f["n_alias_tset"] = al
    a1, ap = s1["addr"].values[i1], pool["addr"].values[ip]
    f["a_tset"] = _cp(a1.tolist(), ap.tolist(), fuzz.token_set_ratio)
    common, first_eq, _, _, _ = _num_feats(ctx.s1_nums[i1], ctx.pool_nums[ip])
    f["num_common"] = common
    f["num_first_eq"] = first_eq
    f["p_addr_empty"] = (ap == "").astype(np.uint8)
    f["p_indic"] = pool["indic"].values[ip].astype(np.uint8)
    f["keymask"] = keymask.astype(np.uint8)
    f["src"] = pool["src"].values[ip].astype(np.uint8)
    f["s1_name_freq"] = ctx.s1_name_freq[i1]
    f["p_name_freq"] = ctx.p_name_freq[ip]
    prod = ctx.A_name[i1].multiply(ctx.B_name[ip]).tocsr()
    shared = np.asarray(prod.sum(1)).ravel()
    f["n_idf_cov1"] = (shared / np.maximum(ctx.s1_name_idf_sum[i1], 1e-6)).astype(np.float32)
    f["n_idf_cov2"] = (shared / np.maximum(ctx.p_name_idf_sum[ip], 1e-6)).astype(np.float32)
    f["n_max_idf"] = prod.max(axis=1).toarray().ravel().astype(np.float32)
    return f


RICH_EXTRA = ["n_jw", "n_partial", "n_ratio_sorted", "n_lev", "n_core2_tset", "n_first_tok_eq",
              "n_ntok1", "n_ntok2", "n_tfidf", "n_legal", "p_name_in_s1",
              "a_partial", "a_tsort", "a_ratio", "a_idf_cov1", "a_idf_cov2", "a_max_idf",
              "num_jac", "n_nums1", "n_nums2", "unit_match", "p_pobox", "a_len1", "a_len2"]


def rich_features(ctx, i1, ip):
    """Extra (more expensive) features for the pruned candidate set; see RICH_EXTRA."""
    f = {}
    s1, pool = ctx.s1, ctx.pool
    c1, cp = s1["core"].values[i1], pool["core"].values[ip]
    c1l, cpl = c1.tolist(), cp.tolist()
    f["n_jw"] = _cp(c1l, cpl, JaroWinkler.normalized_similarity, np.float32)
    f["n_partial"] = _cp(c1l, cpl, fuzz.partial_ratio)
    f["n_ratio_sorted"] = _cp(s1["sorted_core"].values[i1].tolist(), pool["sorted_core"].values[ip].tolist(), fuzz.ratio)
    f["n_lev"] = _cp(c1l, cpl, Levenshtein.normalized_similarity, np.float32)
    f["n_core2_tset"] = _cp([ctx.s1_core2[i] for i in i1], [ctx.p_core2[i] for i in ip], fuzz.token_set_ratio)
    first1 = np.array([c.split(" ", 1)[0] for c in c1l], dtype=object)
    firstp = np.array([c.split(" ", 1)[0] for c in cpl], dtype=object)
    f["n_first_tok_eq"] = (first1 == firstp).astype(np.uint8)
    f["n_ntok1"] = np.diff(ctx.A_name.indptr)[i1].astype(np.uint8)
    f["n_ntok2"] = np.diff(ctx.B_name.indptr)[ip].astype(np.uint8)
    if ctx.tfidf is not None:
        f["n_tfidf"] = _rowdot(ctx.T1, ctx.TP, i1, ip).astype(np.float32)
    else:
        f["n_tfidf"] = np.full(len(i1), np.nan, np.float32)
    l1, lp = s1["legal"].values[i1], pool["legal"].values[ip]
    f["n_legal"] = np.where((l1 == "") | (lp == ""), 2, (l1 == lp).astype(np.int8)).astype(np.int8)
    f["p_name_in_s1"] = ctx.p_name_in_s1[ip]
    a1, ap = s1["addr"].values[i1].tolist(), pool["addr"].values[ip].tolist()
    f["a_partial"] = _cp(a1, ap, fuzz.partial_ratio)
    f["a_tsort"] = _cp(a1, ap, fuzz.token_sort_ratio)
    f["a_ratio"] = _cp(a1, ap, fuzz.ratio)
    shared = _rowdot(ctx.A_addr, ctx.B_addr, i1, ip)
    f["a_idf_cov1"] = (shared / np.maximum(ctx.s1_addr_idf_sum[i1], 1e-6)).astype(np.float32)
    f["a_idf_cov2"] = (shared / np.maximum(ctx.p_addr_idf_sum[ip], 1e-6)).astype(np.float32)
    f["a_max_idf"] = _rowmax(ctx.A_addr, ctx.B_addr, i1, ip).astype(np.float32)
    _, _, jac, nn1, nn2 = _num_feats(ctx.s1_nums[i1], ctx.pool_nums[ip])
    f["num_jac"], f["n_nums1"], f["n_nums2"] = jac, nn1, nn2
    u1, up = s1["unit"].values[i1], pool["unit"].values[ip]
    f["unit_match"] = np.where((u1 == "") | (up == ""), 2, (u1 == up).astype(np.int8)).astype(np.int8)
    f["p_pobox"] = (pool["pobox"].values[ip] != "").astype(np.uint8)
    f["a_len1"] = np.diff(ctx.A_addr.indptr)[i1].astype(np.uint8)
    f["a_len2"] = np.diff(ctx.B_addr.indptr)[ip].astype(np.uint8)
    return f
