import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth

from src.core.settings import Settings, get_settings


class WeaviateClient:
    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()

        self.url = settings.weaviate_url
        self.api_key = settings.weaviate_api_key

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
        )

    def close(self):
        self.client.close()

    def create_collection(self, 
                          collection_name: str, 
                          description: str | None = None):
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