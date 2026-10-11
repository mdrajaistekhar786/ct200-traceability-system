from app.database import requirements_collection
from app.agents.vector_store import index_document_requirements

for document_id in requirements_collection.distinct("document_id"):
    count = index_document_requirements(document_id)
    print(f"{document_id} -> {count} vectors indexed")