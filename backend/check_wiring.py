from app.config import settings
from app.services.rms_service import RMSService

print(f"VECTOR_STORE_TYPE from settings: {settings.VECTOR_STORE_TYPE}")

service = RMSService()
print(f"Actual vector_store class in use: {type(service.policy_retriever.vector_store).__name__}")