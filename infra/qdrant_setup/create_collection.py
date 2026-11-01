"""
Person A (S) -- infra/qdrant_setup/create_collection.py

Creates the sharded Qdrant collection used for the whole project.
Run this once after `docker-compose up -d`.
"""

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def load_config(path="configs/cluster_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_collection(client: QdrantClient, config: dict):
    distance_map = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT,
    }
    client.recreate_collection(
        collection_name=config["collection_name"],
        vectors_config=VectorParams(
            size=config["vector_size"],
            distance=distance_map[config["distance_metric"]],
        ),
        shard_number=config["shards"]["count"],
        replication_factor=config["shards"]["replication_factor"],
    )
    print(
        f"Created collection '{config['collection_name']}' with "
        f"{config['shards']['count']} shards, "
        f"replication factor {config['shards']['replication_factor']}."
    )


if __name__ == "__main__":
    cfg = load_config()
    for node in cfg["nodes"]:
        client = QdrantClient(host=node["host"], port=node["port"])
        print(f"Creating collection on {node['name']} ({node['host']}:{node['port']})...")
        create_collection(client, cfg)