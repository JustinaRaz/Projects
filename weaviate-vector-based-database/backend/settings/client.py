import os

import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth

load_dotenv()


class WeaviateClient:
    def __init__(self):
        self.url = os.environ["WEAVIATE_URL"]
        self.api_key = os.environ["WEAVIATE_API_KEY"]

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
        )

    def close(self):
        self.client.close()
        
    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collections.exists(collection_name)

    def create_collection(self, collection_name, description: str | None = None):
        return self.client.collections.create(
            name=collection_name,
            description=description,
            vector_config=Configure.Vectors.self_provided()
        )
        
    def get_collection(self, collection_name):
        return self.client.collections.use(collection_name)

    def delete_collection(self, collection_name):
        self.client.collections.delete(collection_name)

    def list_collections(self):
        return self.client.collections.list_all()
