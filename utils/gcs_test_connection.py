from google.cloud import storage

client = storage.Client()
buckets = list(client.list_buckets())
print("Buckets accessible:", [bucket.name for bucket in buckets])
