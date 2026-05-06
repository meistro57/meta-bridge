#!/usr/bin/env python3
"""
cluster_qdrant.py
-----------------
Pulls all vectors from the chatbridge_core Qdrant collection,
runs HDBSCAN clustering (falls back to MiniBatchKMeans if hdbscan
is not installed), then writes the cluster label back to every
point as  payload["cluster"] = <int>.

After running, go to the Qdrant visualiser and use:
  {
    "limit": 3000,
    "color_by": { "payload": "cluster" },
    "algorithm": "UMAP"
  }

Requirements (install once):
  pip install qdrant-client numpy scikit-learn hdbscan umap-learn tqdm
"""

import argparse
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointIdsList, SetPayload

# ── config ────────────────────────────────────────────────────────────────────
QDRANT_URL   = "http://localhost:6333"
COLLECTION   = "chats"
BATCH_SIZE   = 500          # points per scroll page
N_CLUSTERS   = 15           # used only for KMeans fallback
HDBSCAN_MIN  = 30           # hdbscan min_cluster_size
PCA_DIMS     = 50           # reduce dims before clustering for speed
UMAP_DIMS    = 10           # optional UMAP pre-reduction (set 0 to skip)
# ─────────────────────────────────────────────────────────────────────────────


def scroll_all(client: QdrantClient):
    """Scroll through the entire collection and return (ids, vectors)."""
    ids, vecs = [], []
    offset = None
    print("Fetching vectors from Qdrant …")
    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_vectors=True,
            with_payload=False,
        )
        if not result:
            break
        for p in result:
            ids.append(p.id)
            vecs.append(p.vector)
        print(f"  fetched {len(ids):,} points …", end="\r")
        if offset is None:
            break
    print(f"\nTotal: {len(ids):,} points")
    return ids, np.array(vecs, dtype=np.float32)


def reduce_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    print(f"PCA → {n_components} dims …")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)


def reduce_umap(X: np.ndarray, n_components: int) -> np.ndarray:
    try:
        import umap
        print(f"UMAP → {n_components} dims …")
        reducer = umap.UMAP(n_components=n_components, random_state=42,
                            metric="cosine", verbose=True)
        return reducer.fit_transform(X)
    except ImportError:
        print("umap-learn not installed, skipping UMAP pre-reduction.")
        return X


def cluster_hdbscan(X: np.ndarray):
    import hdbscan
    print(f"HDBSCAN clustering (min_cluster_size={HDBSCAN_MIN}) …")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN,
                                 metric="euclidean",
                                 core_dist_n_jobs=-1)
    labels = clusterer.fit_predict(X)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    print(f"  → {n} clusters, {noise:,} noise points (label=-1)")
    return labels


def cluster_kmeans(X: np.ndarray):
    from sklearn.cluster import MiniBatchKMeans
    print(f"MiniBatchKMeans clustering (k={N_CLUSTERS}) …")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42,
                         batch_size=4096, n_init=5, verbose=1)
    labels = km.fit_predict(X)
    print(f"  → {N_CLUSTERS} clusters")
    return labels


def write_clusters(client: QdrantClient, ids: list, labels: np.ndarray):
    print("Writing cluster labels back to Qdrant …")
    # group by label to minimise API calls
    from collections import defaultdict
    groups = defaultdict(list)
    for pid, lbl in zip(ids, labels):
        groups[int(lbl)].append(pid)

    for lbl, pids in tqdm(groups.items(), desc="Uploading"):
        # upload in chunks of 1000
        for i in range(0, len(pids), 1000):
            chunk = pids[i:i+1000]
            client.set_payload(
                collection_name=COLLECTION,
                payload={"cluster": lbl},
                points=chunk,
            )
    print("Done!")


def main():
    global N_CLUSTERS, PCA_DIMS, UMAP_DIMS, COLLECTION

    parser = argparse.ArgumentParser(description="Cluster chatbridge_core vectors")
    parser.add_argument("--url",        default=QDRANT_URL)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--method",     default="auto",
                        choices=["auto", "hdbscan", "kmeans"])
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--pca-dims",   type=int, default=PCA_DIMS)
    parser.add_argument("--umap-dims",  type=int, default=UMAP_DIMS)
    args = parser.parse_args()

    N_CLUSTERS = args.n_clusters
    PCA_DIMS   = args.pca_dims
    UMAP_DIMS  = args.umap_dims
    COLLECTION = args.collection

    client = QdrantClient(url=args.url)

    ids, X = scroll_all(client)

    if PCA_DIMS and PCA_DIMS < X.shape[1]:
        X = reduce_pca(X, PCA_DIMS)

    if UMAP_DIMS and UMAP_DIMS < X.shape[1]:
        X = reduce_umap(X, UMAP_DIMS)

    # pick clustering method
    method = args.method
    if method == "auto":
        try:
            import hdbscan as _  # noqa
            method = "hdbscan"
        except ImportError:
            method = "kmeans"

    if method == "hdbscan":
        labels = cluster_hdbscan(X)
    else:
        labels = cluster_kmeans(X)

    write_clusters(client, ids, labels)

    print("\n✅  All done!  Now run the visualiser with:")
    print('   { "limit": 3000, "color_by": { "payload": "cluster" }, "algorithm": "UMAP" }')


if __name__ == "__main__":
    main()
