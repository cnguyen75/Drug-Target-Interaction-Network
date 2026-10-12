"""Node2Vec embeddings training pipeline."""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import EMB_ALL_FP, G_BIP_FP

try:
    from node2vec import Node2Vec
except ImportError:
    print("Error: node2vec not installed. Please install with: pip install node2vec")
    sys.exit(1)


def embed_graph(G, dims: int = 128, walklen: int = 40, walks: int = 10, seed: int = 42):
    """Node2Vec embeddings for the non-isolated nodes of G, as a DataFrame indexed by node_id.

    Uses the same settings as train_embeddings (p=q=1, window=10). workers=1 and a
    fixed seed make the walks reproducible; gensim also needs PYTHONHASHSEED set.
    """
    G = G.subgraph([n for n in G.nodes() if G.degree(n) > 0])
    g2v = Node2Vec(G, dimensions=dims, walk_length=walklen, num_walks=walks,
                   workers=1, p=1, q=1, seed=seed, quiet=True)
    model = g2v.fit(window=10, min_count=1, batch_words=4, seed=seed, workers=1)
    node_ids = list(G.nodes())
    vectors = np.vstack([model.wv[str(n)] for n in node_ids])
    df = pd.DataFrame(vectors, index=pd.Index(node_ids, name="node_id"),
                      columns=[f"e{j}" for j in range(dims)])
    return df


def train_embeddings(dims: int = 128, epochs: int = 3, walklen: int = 40, 
                    walks: int = 10, seed: int = 42, g_bip_fp=None, emb_fp=None):
    """
    Train Node2Vec embeddings on the bipartite graph.
    
    Args:
        dims: Embedding dimensions
        epochs: Training epochs
        walklen: Walk length for random walks
        walks: Number of walks per node
        seed: Random seed for reproducibility
    """
    # Use provided paths or default to config paths
    g_bip_fp = g_bip_fp or G_BIP_FP
    emb_fp = emb_fp or EMB_ALL_FP
    
    print(f"Loading bipartite graph from {g_bip_fp}")
    with open(g_bip_fp, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Set random seed
    np.random.seed(seed)
    
    # Train Node2Vec
    print(f"Training Node2Vec embeddings (dims={dims}, epochs={epochs}, "
          f"walklen={walklen}, walks={walks})")
    start_time = time.time()
    
    # Create Node2Vec model
    g2v = Node2Vec(
        G,
        dimensions=dims,
        walk_length=walklen,
        num_walks=walks,
        workers=1,
        p=1,
        q=1,
        seed=seed
    )
    
    # Fit the model
    model = g2v.fit(window=10, min_count=1, batch_words=4)
    
    # Get embeddings for all nodes
    node_ids = list(G.nodes())
    embeddings = np.zeros((len(node_ids), dims))
    
    for i, node_id in enumerate(node_ids):
        if str(node_id) in model.wv:
            embeddings[i] = model.wv[str(node_id)]
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.1f} seconds")
    
    # Create DataFrame with embeddings
    node_ids = list(G.nodes())
    embedding_data = []
    
    for i, node_id in enumerate(node_ids):
        row = {"node_id": node_id}
        for j in range(dims):
            row[f"e{j}"] = embeddings[i, j]
        embedding_data.append(row)
    
    embeddings_df = pd.DataFrame(embedding_data)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings DataFrame: {len(embeddings_df)} rows, {len(embeddings_df.columns)} columns")
    
    # Save embeddings
    embeddings_df.to_parquet(emb_fp, index=False)
    print(f"Saved embeddings to {emb_fp}")
    
    # Print some statistics
    print("\nEmbedding Statistics:")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")


def main():
    """Main function to train embeddings."""
    parser = argparse.ArgumentParser(description="Train Node2Vec embeddings")
    parser.add_argument("--dims", type=int, default=128, help="Embedding dimensions")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--walklen", type=int, default=40, help="Walk length")
    parser.add_argument("--walks", type=int, default=10, help="Number of walks per node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if not G_BIP_FP.exists():
        print("Error: Bipartite graph not found.")
        print("Please run the build_graph step first.")
        sys.exit(1)
    
    train_embeddings(
        dims=args.dims,
        epochs=args.epochs,
        walklen=args.walklen,
        walks=args.walks,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
