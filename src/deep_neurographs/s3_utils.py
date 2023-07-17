"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for reading and writing to an s3 bucket.

"""

import io

import boto3


def init_session(access_key_id=None, secret_access_key=None):
    if access_key_id is None or access_key_id is None:
        session = boto3.Session()
    else:
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
    return session.client("s3")


def listdir(bucket, dir_prefix, s3_client, ext=None):
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=dir_prefix)
    filenames = []
    for file in response["Contents"]:
        file_key = file["Key"]
        if ext is not None:
            if ext in file_key:
                filenames.append(file_key)
        else:
            filenames.append(file_key)
    return filenames


def read_from_s3(bucket, file_key, s3_client):
    if ".txt" in file_key or ".swc" in file_key:
        return read_txt(bucket, file_key, s3_client)
    else:
        assert True, "File type of {} is not supported".format(file_key)


def read_txt(bucket, file_key, s3_client):
    body = []
    s3_object = s3_client.get_object(Bucket=bucket, Key=file_key)
    for line in io.TextIOWrapper(s3_object["Body"]):
        if not line.startswith("#"):
            line = line.replace("\n", "")
            body.append(line)
    return body
