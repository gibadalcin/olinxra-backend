import os
from google.cloud import storage

bucket_name = "olinxra-modelos"
source_blob_name = "quantized_clip_model.onnx"
destination_file_name = "quantized_clip_model.onnx"
cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "cloud-storage-cred.json")

storage_client = storage.Client.from_service_account_json(cred_path)
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
print(f"Modelo baixado para {destination_file_name}")
