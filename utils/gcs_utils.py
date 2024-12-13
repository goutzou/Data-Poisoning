from google.cloud import storage

def upload_to_gcs(dataframe, bucket_name, destination_blob_name):
    """
    Uploads a Pandas DataFrame to Google Cloud Storage as a CSV file.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    csv_data = dataframe.to_csv(index=False)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(csv_data, content_type="text/csv")
    print(f"Uploaded {destination_blob_name} to {bucket_name}.")

def load_from_gcs(bucket_name, blob_name, local_file_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_file_path)
    print(f"Downloaded {blob_name} to {local_file_path}.")

