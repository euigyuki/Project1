import boto3
import csv


def get_s3_urls(bucket_name, region="us-east-1"):
    s3 = boto3.client("s3", region_name=region)

    # List objects in the bucket
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name)

    urls = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{key}"
            urls.append((key, url))

    return urls


# Replace with your bucket name and region
bucket_name = "derrickbucketversion4"
region = "us-east-1"
filename = "s3_urls_version4.csv"
urls = get_s3_urls(bucket_name, region)

# Write URLs to a CSV file
with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Object Key", "image_url"])
    writer.writerows(urls)

print(f"URLs have been written")
