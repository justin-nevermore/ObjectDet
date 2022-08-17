import boto3
import os

s3 = boto3.resource('s3')
folder_uri = "s3://labels-v3-datasets-499359467031-us-west-2/LabelsDataDelivery/LabelsV3/BatchE-Detection/truck/"
bucket, key = folder_uri[5:].split('/', 1)
bucket_obj = s3.Bucket(bucket)
limit = 200
count = 0
for obj_data in bucket_obj.objects.filter(Prefix=key):
    uri = obj_data.key
    filename = os.path.basename(uri)
    s3.Bucket(bucket).download_file(uri, f'test_images/{filename}')
    count += 1
    if count == limit:
        break