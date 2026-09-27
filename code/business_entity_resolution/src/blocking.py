"""Stage 1a: candidate generation (blocking) inside one country.

Union of complementary keys, joined with vectorised sort/searchsorted expansion
(no Python loops over pairs). Every key has a block-size cap so generic values
never explode:

  BIT_NAME  exact sorted-core name (pool: name and alias)
  BIT_TOK   single rare name token                              (typo in all other words)
  BIT_TOK2  pair of name tokens among each record's rarest ones  ("good|tech": rare as a pair)
  BIT_SKEL  pair of skeleton tokens (pool side: Indic-script names only) -> transliterations
  BIT_ADDR  (house number, address word)                         -> coined aliases, domains
  BIT_NUMT  (house number, name token)                           -> address words differ

Meta-blocking: every matched key contributes a weight 1/log2(2 + block size);
pairs are ranked by the summed weight and only the best `keep_per_src`
candidates per (S1, source) survive to feature computation.
"""
import numpy as np

BIT_NAME, BIT_TOK, BIT_TOK2, BIT_SKEL, BIT_ADDR, BIT_NUMT, BIT_NOSP, BIT_AW2 = 1, 2, 4, 8, 16, 32, 64, 128
KEY_BITS = [(BIT_NAME, "k_name"), (BIT_TOK, "k_tok"), (BIT_TOK2, "k_tok2"), (BIT_SKEL, "k_skel"),
            (BIT_ADDR, "k_addr"), (BIT_NUMT, "k_numt"), (BIT_NOSP, "k_nosp"), (BIT_AW2, "k_aw2")]


class Index:
    """Sorted multimap key -> rows for the pool side."""

    def __init__(self, rows, keys):
        order = np.argsort(keys, kind="stable")
        k = keys[order]
        self.rows = rows[order]
        self.uk, self.start, self.cnt = np.unique(k, return_index=True, return_counts=True)

    def query(self, q_rows, q_keys, max_block):
        empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))
        if len(self.uk) == 0 or len(q_keys) == 0:
            return empty
        pos = np.searchsorted(self.uk, q_keys)
        pos = np.minimum(pos, len(self.uk) - 1)
        ok = (self.uk[pos] == q_keys) & (self.cnt[pos] <= max_block)
        pos, qr = pos[ok], q_rows[ok]
        if len(pos) == 0:
            return empty
        reps = self.cnt[pos]
        total = int(reps.sum())
        out_q = np.repeat(qr, reps)
        base = np.repeat(self.start[pos] - (np.cumsum(reps) - reps), reps)
        out_p = self.rows[base + np.arange(total)]
        w = np.repeat((1.0 / np.log2(2.0 + reps)).astype(np.float32), reps)
        return out_q, out_p, w


def to_csr(strings, vocab):
    """Space-separated token strings -> (indptr, ids) with a shared, growing vocab."""
    indptr = np.zeros(len(strings) + 1, dtype=np.int64)
    ids = []
    get = vocab.get
    for i, s in enumerate(strings):
        if s:
            for t in s.split():
                v = get(t)
                if v is None:
                    v = len(vocab)
                    vocab[t] = v
                ids.append(v)
        indptr[i + 1] = len(ids)
    return indptr, np.asarray(ids, dtype=np.int64)


def csr_rows(indptr):
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))


def rarest_per_row(rows, toks, df, k, max_df=None):
    """Keep the k rarest distinct tokens (df <= max_df) of every row, rarest first."""
    d = df[toks]
    if max_df is not None:
        ok = d <= max_df
        rows, toks, d = rows[ok], toks[ok], d[ok]
    if len(rows) == 0:
        return rows, toks
    order = np.lexsort((toks, d, rows))
    rows, toks = rows[order], toks[order]
    dup = np.r_[False, (rows[1:] == rows[:-1]) & (toks[1:] == toks[:-1])]
    rows, toks = rows[~dup], toks[~dup]
    first = np.r_[True, rows[1:] != rows[:-1]]
    start_idx = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    rank = np.arange(len(rows)) - start_idx[grp]
    keep = rank < k
    return rows[keep], toks[keep]


def pair_keys(rows, toks, vocab_size):
    """All unordered token pairs within each row (rows sorted, <= k tokens/row) -> int64 keys."""
    if len(rows) == 0:
        return rows, rows
    first = np.r_[True, rows[1:] != rows[:-1]]
    start = np.flatnonzero(first)
    size = np.diff(np.r_[start, len(rows)])
    out_r, out_k = [], []
    kmax = int(size.max())
    for a in range(kmax):
        for b in range(a + 1, kmax):
            ok = size > b
            ia, ib = start[ok] + a, start[ok] + b
            ta, tb = np.minimum(toks[ia], toks[ib]), np.maximum(toks[ia], toks[ib])
            out_r.append(rows[ia])
            out_k.append(ta * vocab_size + tb)
    return np.concatenate(out_r), np.concatenate(out_k)


def cross_keys(rows_a, toks_a, nums_mat, n_nums):
    """(token, number) keys: every kept token of a row x its first n_nums numbers."""
    r_all, k_all = [], []
    for j in range(min(n_nums, nums_mat.shape[1])):
        nv = nums_mat[rows_a, j]
        ok = nv >= 0
        r_all.append(rows_a[ok])
        k_all.append((toks_a[ok] << 31) | nv[ok])
    if not r_all:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(r_all), np.concatenate(k_all)


class CountryBlocker:
    """Builds pool-side indexes once; S1 row chunks are then blocked against them."""

    DEFAULT = {"name_cap": 300, "tok_df": 300, "tok_k": 3, "tok2_k": 4, "tok2_cap": 300,
               "skel_k": 4, "skel_cap": 300, "addr_k_s1": 8, "addr_k_pool": 8, "addr_cap": 300,
               "numt_k": 2, "numt_cap": 150, "aw2_k": 4, "aw2_cap": 300, "keep_per_src": 15,
               "pre_cap": 450}

    def __init__(self, s1, s1_nums, pool, pool_nums, cfg=None):
        self.cfg = c = {**self.DEFAULT, **(cfg or {})}
        self.n1, self.npool = len(s1), len(pool)
        self.pool_src = pool["src"].values.astype(np.int64)
        self.s1_core, self.p_core = s1["core"].values, pool["core"].values
        self.s1_addr, self.p_addr = s1["addr"].values, pool["addr"].values
        self.s1_skel, self.p_skel = s1["skel"].values, pool["skel"].values
        self.s1_nospace, self.p_nospace = s1["nospace"].values, pool["nospace"].values
        self.p_indic = pool["indic"].values.astype(bool)
        import os
        import pickle
        from config import work_path
        rp = work_path("ranker.pkl")
        if os.path.exists(rp) and not (cfg or {}).get("no_ranker"):
            with open(rp, "rb") as fh:
                self.ranker = pickle.load(fh)
        prow = np.arange(self.npool, dtype=np.int64)
        s1_parts, self.indexes = {}, {}

        # exact sorted-core names (pool: name + alias) -----------------------------------
        nv = {}
        s1k = np.array([nv.setdefault(s, len(nv)) if s else -1 for s in s1["sorted_core"].values], np.int64)
        pk = np.array([nv.setdefault(s, len(nv)) if s else -1 for s in pool["sorted_core"].values], np.int64)
        pa = np.array([nv.setdefault(s, len(nv)) if s else -1 for s in pool["alias_sorted"].values], np.int64)
        self.indexes[BIT_NAME] = (Index(np.r_[prow[pk >= 0], prow[pa >= 0]], np.r_[pk[pk >= 0], pa[pa >= 0]]),
                                  c["name_cap"])
        r1 = np.arange(self.n1, dtype=np.int64)
        s1_parts[BIT_NAME] = (r1[s1k >= 0], s1k[s1k >= 0])

        # exact no-space name ("hariomfood.com" ~ "Hariom Food") ------------------------------
        nsv = {}
        s1n = np.array([nsv.setdefault(s, len(nsv)) if len(s) >= 4 else -1 for s in s1["nospace"].values], np.int64)
        pn = np.array([nsv.setdefault(s, len(nsv)) if len(s) >= 4 else -1 for s in pool["nospace"].values], np.int64)
        self.indexes[BIT_NOSP] = (Index(prow[pn >= 0], pn[pn >= 0]), c["name_cap"])
        s1_parts[BIT_NOSP] = (r1[s1n >= 0], s1n[s1n >= 0])

        # name tokens -----------------------------------------------------------------------
        tv = {}
        s1_ip, s1_ids = to_csr(s1["core"].values, tv)
        p_ip, p_ids = to_csr((pool["core"] + " " + pool["alias"]).values, tv)
        V = len(tv)
        df = np.bincount(s1_ids, minlength=V) + np.bincount(p_ids, minlength=V)
        s1_r, p_r = csr_rows(s1_ip), csr_rows(p_ip)
        # single rare token
        pr1, pt1 = rarest_per_row(p_r, p_ids, df, 99, c["tok_df"])
        self.indexes[BIT_TOK] = (Index(pr1, pt1), c["tok_df"])
        s1_parts[BIT_TOK] = rarest_per_row(s1_r, s1_ids, df, c["tok_k"], c["tok_df"])
        # token pairs among the rarest tokens
        pr2, pt2 = rarest_per_row(p_r, p_ids, df, c["tok2_k"])
        self.indexes[BIT_TOK2] = (Index(*pair_keys(pr2, pt2, V)), c["tok2_cap"])
        sr2, st2 = rarest_per_row(s1_r, s1_ids, df, c["tok2_k"])
        s1_parts[BIT_TOK2] = pair_keys(sr2, st2, V)

        # skeleton token pairs (pool: Indic-script names only) ---------------------------------
        sv = {}
        s1_sp, s1_sids = to_csr(s1["skel"].values, sv)
        p_skel = np.where(pool["indic"].values == 1, pool["skel"].values, "")
        p_sp, p_sids = to_csr(p_skel, sv)
        sdf = np.bincount(s1_sids, minlength=len(sv)) + np.bincount(p_sids, minlength=len(sv))
        psr, pst = rarest_per_row(csr_rows(p_sp), p_sids, sdf, c["skel_k"])
        self.indexes[BIT_SKEL] = (Index(*pair_keys(psr, pst, len(sv))), c["skel_cap"])
        ssr, sst = rarest_per_row(csr_rows(s1_sp), s1_sids, sdf, c["skel_k"])
        s1_parts[BIT_SKEL] = pair_keys(ssr, sst, len(sv))

        # (house number, address word) ------------------------------------------------------
        wv = {}
        words = lambda arr: [" ".join(t for t in a.split() if not t.isdigit()) if a else "" for a in arr]
        s1_wp, s1_wids = to_csr(words(s1["addr"].values), wv)
        p_wp, p_wids = to_csr(words(pool["addr"].values), wv)
        wdf = np.bincount(s1_wids, minlength=len(wv)) + np.bincount(p_wids, minlength=len(wv))
        prw, ptw = rarest_per_row(csr_rows(p_wp), p_wids, wdf, c["addr_k_pool"])
        self.indexes[BIT_ADDR] = (Index(*cross_keys(prw, ptw, pool_nums, 3)), c["addr_cap"])
        srw, stw = rarest_per_row(csr_rows(s1_wp), s1_wids, wdf, c["addr_k_s1"])
        s1_parts[BIT_ADDR] = cross_keys(srw, stw, s1_nums, 3)
        del prw, ptw
        # pairs of rare address words (addresses without house numbers)
        pr4, pt4 = rarest_per_row(csr_rows(p_wp), p_wids, wdf, c["aw2_k"])
        self.indexes[BIT_AW2] = (Index(*pair_keys(pr4, pt4, len(wv))), c["aw2_cap"])
        sr4, st4 = rarest_per_row(csr_rows(s1_wp), s1_wids, wdf, c["aw2_k"])
        s1_parts[BIT_AW2] = pair_keys(sr4, st4, len(wv))

        # (house number, name token) --------------------------------------------------------
        prn, ptn = rarest_per_row(p_r, p_ids, df, 4)
        self.indexes[BIT_NUMT] = (Index(*cross_keys(prn, ptn, pool_nums, 3)), c["numt_cap"])
        srn, stn = rarest_per_row(s1_r, s1_ids, df, c["numt_k"])
        s1_parts[BIT_NUMT] = cross_keys(srn, stn, s1_nums, 2)

        # sort every S1 key list by row once, so chunks are contiguous slices
        self.s1_parts = {}
        for bit, (r, k) in s1_parts.items():
            o = np.argsort(r, kind="stable")
            self.s1_parts[bit] = (r[o], k[o])

    stats = None
    truth = None        # optional sorted int64 codes of true pairs (dev diagnostics only)
    dump = None         # optional list collecting ranker training data (dev runs only)

    def _stat(self, key, val):
        if self.stats is None:
            self.stats = {}
        self.stats[key] = self.stats.get(key, 0) + val

    def _hits(self, key, codes):
        if self.truth is not None:
            self._stat(key, int(np.isin(codes, self.truth).sum()))

    def block(self, lo, hi, rows_mask=None):
        """Candidates for S1 rows [lo, hi) (optionally restricted by a boolean row mask).
        Returns i1, ip, keymask (uint8), meta-score (float32), after the per-source cap."""
        import time
        t0 = time.time()
        cod, bits, wts = [], [], []
        for bit, (r, k) in self.s1_parts.items():
            a, b = np.searchsorted(r, lo), np.searchsorted(r, hi)
            rr, kk = r[a:b], k[a:b]
            if rows_mask is not None:
                m = rows_mask[rr]
                rr, kk = rr[m], kk[m]
            idx, cap = self.indexes[bit]
            q, p, w = idx.query(rr, kk, cap)
            cod.append(q * self.npool + p)
            bits.append(np.full(len(q), bit, np.uint8))
            wts.append(w)
        codes = np.concatenate(cod)
        if len(codes) == 0:
            e = np.empty(0, np.int64)
            return e, e, np.empty(0, np.uint8), np.empty(0, np.float32)
        bits, wts = np.concatenate(bits), np.concatenate(wts)
        t1 = time.time()
        order = np.argsort(codes)
        codes, bits, wts = codes[order], bits[order], wts[order]
        first = np.r_[True, codes[1:] != codes[:-1]]
        starts = np.flatnonzero(first)
        mask = np.bitwise_or.reduceat(bits, starts)
        score = np.add.reduceat(wts, starts)
        uc = codes[starts]
        self._stat("n_union", len(uc))
        self._hits("hit_union", uc)
        i1, ip = uc // self.npool, uc % self.npool
        # cheap pre-cap on the key-evidence score before any string scoring
        pre = self._top_per_group(i1 - lo, score, self.cfg["pre_cap"])
        i1, ip, mask, score = i1[pre], ip[pre], mask[pre], score[pre]
        self._stat("n_pre", len(i1))
        self._hits("hit_pre", i1 * self.npool + ip)
        t2 = time.time()
        if self.dump is not None:                    # ranker training data (dev runs only)
            f = self.quick_features(i1, ip, score, mask)
            f = {k: v.astype(np.float32) for k, v in f.items()}
            f["label"] = np.isin(i1 * self.npool + ip, self.truth).astype(np.int8)
            f["i1"] = i1.astype(np.int32)
            self.dump.append(f)
        quick = self.quick_score(i1, ip, score, mask)
        t3 = time.time()
        self._stat("t_query", t1 - t0)
        self._stat("t_sort", t2 - t1)
        self._stat("t_quick", t3 - t2)
        # keep the best `keep_per_src` per (S1, source) by the quick similarity
        keep = self._top_per_group((i1 - lo) * 4 + self.pool_src[ip], quick, self.cfg["keep_per_src"])
        return i1[keep], ip[keep], mask[keep], score[keep], quick[keep]

    @staticmethod
    def _top_per_group(grp, score, k):
        """Indices (sorted) of the k best rows per group, via one int64 argsort on a packed key."""
        s = score.astype(np.float64)
        q = np.round((s.max() - s) / max(s.max() - s.min(), 1e-9) * (2 ** 30)).astype(np.int64)
        o = np.argsort((grp.astype(np.int64) << 31) | q)
        g = grp[o]
        f = np.r_[True, g[1:] != g[:-1]]
        rank = np.arange(len(g)) - np.flatnonzero(f)[np.cumsum(f) - 1]
        return np.sort(o[rank < k])

    ranker = None       # optional learned model on quick_features (see train_ranker.py)
    RANK_FEATS = ["q_name", "q_addr", "q_ns", "q_meta", "q_indic", "q_paddr_empty", "q_src"] + \
                 [n for _, n in KEY_BITS]

    def quick_features(self, i1, ip, meta, mask):
        from rapidfuzz import fuzz
        from rapidfuzz.process import cpdist
        n = cpdist(self.s1_core[i1].tolist(), self.p_core[ip].tolist(), scorer=fuzz.token_set_ratio,
                   workers=-1, dtype=np.float32)
        ind = self.p_indic[ip]
        if ind.any():
            s = cpdist(self.s1_skel[i1[ind]].tolist(), self.p_skel[ip[ind]].tolist(),
                       scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            n[ind] = np.maximum(n[ind], s)
        a = cpdist(self.s1_addr[i1].tolist(), self.p_addr[ip].tolist(), scorer=fuzz.token_set_ratio,
                   workers=-1, dtype=np.float32)
        f = {"q_name": n, "q_addr": a, "q_meta": meta.astype(np.float32),
             "q_indic": ind.astype(np.float32), "q_paddr_empty": (self.p_addr[ip] == "").astype(np.float32),
             "q_src": self.pool_src[ip].astype(np.float32)}
        if self.ranker is not None or self.dump is not None:
            f["q_ns"] = cpdist(self.s1_nospace[i1].tolist(), self.p_nospace[ip].tolist(), scorer=fuzz.ratio,
                               workers=-1, dtype=np.float32)
            for bit, name in KEY_BITS:
                f[name] = ((mask & bit) > 0).astype(np.float32)
        return f

    def quick_score(self, i1, ip, meta, mask=None):
        """Orders the union before the per-source cap. Default: max(name, skeleton-name)
        token-set ratio + 0.8 * address token-set ratio + key evidence; with a trained
        ranker: its predicted match probability on the same fast features."""
        f = self.quick_features(i1, ip, meta, mask if mask is not None else np.zeros(len(i1), np.uint8))
        if self.ranker is not None:
            X = np.column_stack([f[k] for k in self.RANK_FEATS]).astype(np.float32)
            return self.ranker.predict_np(X).astype(np.float32)
        return f["q_name"] + 0.8 * f["q_addr"] + 20.0 * np.minimum(meta, 3.0)
