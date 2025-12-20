from typing import List
import weaviate
from weaviate.classes.config import Property, DataType, Configure

class VectorDb:
    def __init__(self, host: str, http_port: int, grpc_port: int, collection_name: str) -> None:
        self.collection_name = collection_name
        self.client = weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)
        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready. Is Docker running?")

        self.collection = self._ensure_collection()

    def close(self) -> None:
        self.client.close()

    def _ensure_collection(self):
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        return self.client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="note_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.self_provided(),
        )

    def is_empty(self) -> bool:
        info = self.collection.aggregate.over_all(total_count=True)
        return info.total_count == 0

    def insert(self, notes: List[dict], vectors: List[List[float]]) -> None:
        with self.collection.batch.fixed_size(batch_size=16) as batch:
            for rec, vec in zip(notes, vectors):
                batch.add_object(
                    properties={
                        "note_id": rec["note_id"],
                        "title": rec["title"],
                        "content": rec["content"],
                    },
                    vector=vec,
                )

    def search(self, vector: List[float], k: int):
        return self.collection.query.near_vector(near_vector=vector, limit=k)
