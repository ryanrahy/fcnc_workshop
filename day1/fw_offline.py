"""Offline stand-in for `fafbseg.flywire`, used by the workshop notebooks.

    import fw_offline as flywire

answers the same function calls as `from fafbseg import flywire`, but from the
pre-downloaded data in the `fw_data/` folder (created by prepare_offline_data.py)
instead of the FlyWire servers. No FlyWire token is needed. All data are from
the public materialization 783.

Only neurons in the bundle can be queried. For anything else, an
`OfflineDataError` explains what is available.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import navis
import numpy as np
import pandas as pd

DATA = Path(os.environ.get("FW_DATA", Path(__file__).resolve().parent / "fw_data"))
NT_COLS = ["gaba", "acetylcholine", "glutamate", "octopamine", "serotonin", "dopamine"]


class OfflineDataError(LookupError):
    """Raised for queries that are not covered by the offline data bundle."""


# ---------------------------------------------------------------- loading

def _read(name):
    try:
        df = pd.read_parquet(DATA / f"{name}.parquet")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Offline data not found in {DATA}. Put the `fw_data` folder next to "
            "the notebooks (or set the FW_DATA environment variable).") from None
    # back to plain object columns, like fafbseg returns them
    for c in df.columns:
        if isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


@lru_cache(None)
def _meta():
    return json.loads((DATA / "meta.json").read_text())


@lru_cache(None)
def _annotations():
    return _read("annotations")


@lru_cache(None)
def _nodes():
    nodes = _read("skeleton_nodes")
    return {rid: df.drop(columns="root_id") for rid, df in nodes.groupby("root_id")}


@lru_cache(None)
def _synapses():
    return _read("synapses")


@lru_cache(None)
def _connectivity():
    return _read("connectivity")


@lru_cache(None)
def _connectors():
    return _read("gf_connectors")


def available_types():
    """Cell types whose skeletons and connectivity are in the offline bundle."""
    return list(_meta()["explorer_types"])


def _ids(x):
    if isinstance(x, NeuronCriteria):
        return x.get_roots()
    if isinstance(x, navis.NeuronList):
        return np.asarray(x.id, dtype=np.int64)
    if isinstance(x, navis.BaseNeuron):
        return np.array([x.id], dtype=np.int64)
    return np.atleast_1d(np.asarray(x, dtype=np.int64))


def _require(ids, covered, what):
    missing = np.setdiff1d(ids, np.asarray(covered, dtype=np.int64))
    if len(missing):
        raise OfflineDataError(
            f"{len(missing)} of {len(ids)} neuron(s) have no {what} in the offline data. "
            f"Available cell types: {', '.join(available_types())}.")


# ------------------------------------------------------- setup / versions

def set_chunkedgraph_secret(token=None, **kwargs):
    print("Offline mode: no token needed.")


def set_default_dataset(dataset):
    print(f'Default dataset set to "{dataset}" (offline copy of materialization '
          f'{_meta()["materialization"]}).')


def get_materialization_versions(**kwargs):
    return _read("materialization_versions")


# ----------------------------------------------------------- root IDs

def locs_to_segments(locs, timestamp=None, **kwargs):
    out = []
    for loc in np.asarray(locs).reshape(-1, 3):
        key = f"{int(loc[0])},{int(loc[1])},{int(loc[2])}|{timestamp}"
        if key not in _meta()["locs"]:
            raise OfflineDataError(
                f"Location {list(map(int, loc))} (timestamp={timestamp}) is not in the "
                "offline data. Only the locations used in the notebooks are available.")
        out.append(_meta()["locs"][key])
    return np.array(out, dtype=np.int64)


def _is_known_783(ids):
    return np.isin(ids, _annotations().root_id.values)


def is_latest_root(id, **kwargs):
    ids = _ids(id)
    cached = _meta()["is_latest"]
    out = []
    for i in ids:
        if str(i) in cached:
            out.append(cached[str(i)])
        elif _is_known_783([i])[0]:
            out.append(True)  # the public dataset is frozen at v783
        else:
            raise OfflineDataError(f"Root ID {i} is not in the offline data.")
    return np.array(out)


def update_ids(id, **kwargs):
    ids = _ids(id)
    cached = _read("update_ids").set_index("old_id")
    rows = []
    for i in ids:
        if i in cached.index:
            rows.append(cached.loc[[i]].reset_index())
        elif _is_known_783([i])[0]:
            rows.append(pd.DataFrame({"old_id": [i], "new_id": [i], "confidence": [1.0],
                                      "changed": [False]}))
        else:
            raise OfflineDataError(f"Root ID {i} is not in the offline data.")
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------ neurons / plotting

def get_mesh_neuron(id, **kwargs):
    ids = _ids(id)
    if len(ids) != 1 or ids[0] != _meta()["gf_left"]:
        raise OfflineDataError("Only the left giant fiber's mesh is in the offline data.")
    with np.load(DATA / "mesh_gf_left.npz") as f:
        return navis.MeshNeuron((f["vertices"], f["faces"]), id=int(ids[0]), units="1 nm")


def _skeleton(rid):
    nodes = _nodes().get(rid)
    if nodes is None:
        return None
    return navis.TreeNeuron(nodes.copy(), id=rid, units="1 nm", name="skeleton")


def get_skeletons(root_id, omit_failures=None, **kwargs):
    single = isinstance(root_id, (int, np.integer))
    ids = _ids(root_id)
    found = [s for s in (_skeleton(i) for i in ids) if s is not None]
    n_missing = len(ids) - len(found)
    if n_missing and (omit_failures is None or not found):
        raise OfflineDataError(
            f"{n_missing} of {len(ids)} skeleton(s) are not in the offline data. "
            f"Available cell types: {', '.join(available_types())}.")
    if n_missing:
        print(f"(offline: {n_missing} of {len(ids)} skeletons not available, skipped)")
    return found[0] if single else navis.NeuronList(found)


def encode_url(segments=None, **kwargs):
    gfs = {_meta()["gf_left"], _meta()["gf_right"]}
    if segments is not None and not set(_ids(segments)) <= gfs:
        raise OfflineDataError("Offline, only the giant fiber Neuroglancer link is available.")
    return _meta()["ngl_url"]


# ------------------------------------------------------------ annotations

def search_community_annotations(x, **kwargs):
    ca = _read("community_annotations")
    if x is None:
        return ca
    return ca[ca.pt_root_id.isin(_ids(x))].reset_index(drop=True)


def search_annotations(x, exact=False, case=False, regex=True, **kwargs):
    """Search hierarchical annotations (same matching rules as fafbseg)."""
    ann = _annotations()
    if x is None:
        return ann.copy()
    if isinstance(x, NeuronCriteria):
        x = x.get_roots()
    if isinstance(x, str):
        if ":" in x:
            col, x = (s.strip() for s in x.split(":"))
            cols = [col]
        else:
            dt = ann.dtypes
            cols = dt[(dt == object) & ~dt.index.str.contains("root")].index
        keep = np.zeros(len(ann), dtype=bool)
        for col in cols:
            s = ann[col].astype(object).where(ann[col].notna(), None)
            if not exact:
                keep |= s.str.contains(x, case=case, regex=regex, na=False).to_numpy(bool)
            elif not regex:
                keep |= (s == x).to_numpy(bool)
            else:
                keep |= s.str.match(x, case=case, na=False).to_numpy(bool)
        ann = ann.loc[keep]
    else:
        ann = ann[ann.root_id.isin(_ids(x))]
    return ann.copy().reset_index(drop=True)


class NeuronCriteria:
    """Select neurons by annotation fields, e.g. NeuronCriteria(type="LC4", side="left").

    `type` matches both `cell_type` and `hemibrain_type`. Matching is exact
    (case-insensitive) unless `regex=True`.
    """

    def __init__(self, *, regex=False, case=False, verbose=True, **criteria):
        self.criteria, self.regex, self.case, self.verbose = criteria, regex, case, verbose

    def __repr__(self):
        return f"NeuronCriteria({self.criteria})"

    def _filter(self, ann, col, value):
        s = ann[col].astype(object).where(ann[col].notna(), None)
        values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
        if self.regex:
            return ann[s.str.match("|".join(values), case=self.case, na=False).to_numpy(bool)]
        if not self.case:
            return ann[s.str.lower().isin([str(v).lower() for v in values]).to_numpy(bool)]
        return ann[s.isin(values).to_numpy(bool)]

    def get_roots(self):
        ann = _annotations()
        for field, value in self.criteria.items():
            if field == "type":
                ann = pd.concat([self._filter(ann, "cell_type", value),
                                 self._filter(ann, "hemibrain_type", value)]
                                ).drop_duplicates("root_id")
            else:
                ann = self._filter(ann, field, value)
        roots = np.unique(ann.root_id.values).astype(np.int64)
        if not len(roots):
            raise OfflineDataError("No neurons found matching the given criteria.")
        if self.verbose:
            print(f"Found {len(roots)} neuron{'s' if len(roots) > 1 else ''} "
                  "matching the given criteria.")
        return roots


# ------------------------------------------------------ synapses / edges

def get_synapses(x, pre=True, post=True, attach=True, transmitters=False, **kwargs):
    ids = _ids(x)
    meta = _meta()
    old = meta["gf_left_630"]
    if old != meta["gf_left"] and {old, meta["gf_left"]} <= set(ids):
        raise ValueError(meta["mixed_mat_error"] or "IDs never co-existed in one materialization")
    _require(ids, meta["synapse_ids"], "synapses")

    syn = _synapses()
    keep = np.zeros(len(syn), dtype=bool)
    if pre:
        keep |= syn.pre.isin(ids).to_numpy()
    if post:
        keep |= syn.post.isin(ids).to_numpy()
    syn = syn[keep]
    if not transmitters:
        syn = syn.drop(columns=[c for c in NT_COLS if c in syn.columns])
    syn = syn.reset_index(drop=True)

    if attach and isinstance(x, (navis.TreeNeuron, navis.NeuronList)):
        cn = _connectors()
        for n in navis.NeuronList(x):
            c = cn[cn.root_id == np.int64(n.id)].drop(columns="root_id").reset_index(drop=True)
            c["type"] = c["type"].astype("category")
            n.connectors = c
    return syn


def get_connectivity(x, upstream=True, downstream=True, **kwargs):
    if not upstream and not downstream:
        raise ValueError("`upstream` and `downstream` must not both be False")
    ids = _ids(x)
    _require(ids, _meta()["connectivity_ids"], "connectivity")
    cn = _connectivity()
    keep = np.zeros(len(cn), dtype=bool)
    if upstream:
        keep |= cn.post.isin(ids).to_numpy()
    if downstream:
        keep |= cn.pre.isin(ids).to_numpy()
    return cn[keep].sort_values("weight", ascending=False).reset_index(drop=True)


def get_adjacency(sources, targets=None, square=True, **kwargs):
    src = _ids(sources)
    tgt = src if targets is None else _ids(targets)
    covered = np.asarray(_meta()["connectivity_ids"], dtype=np.int64)
    if not (np.isin(src, covered).all() or np.isin(tgt, covered).all()):
        _require(src, covered, "connectivity")
    cn = _connectivity()
    cn = cn[cn.pre.isin(src) & cn.post.isin(tgt)]
    adj = cn.pivot_table(index="pre", columns="post", values="weight", aggfunc="sum", fill_value=0)
    if square:
        src = tgt = np.unique(np.concatenate([src, tgt]))
    adj = adj.reindex(index=src, columns=tgt, fill_value=0).astype(int)
    adj.index.name, adj.columns.name = "source", "target"
    return adj
