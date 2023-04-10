import streamlit as st
import os
import pandas as pd
from io import BytesIO
import boto3
from dotenv import load_dotenv
load_dotenv()

S3_BUCKET = 'writergpt'

def create_s3():
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets['AWS_ACCESS_KEY'],
            aws_secret_access_key=st.secrets['AWS_SECRET_KEY'],
        )
    except FileNotFoundError:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
        )
    return s3

s3 = create_s3()

def load_s3_csv(key, s3 = s3):
    s3_object = s3.get_object(Bucket=S3_BUCKET, Key=key)
    contents = s3_object['Body'].read()
    df = pd.read_csv(BytesIO(contents))
    return df

def save_s3_csv(df, key, s3=s3):
    buffer = BytesIO()
    df.to_csv(buffer)
    s3.put_object(Body=buffer.getvalue(), Bucket=S3_BUCKET, Key=key)
    return True

def get_all_s3_objects(s3=s3, **base_kwargs):
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):  # At the end of the list?
            break
        continuation_token = response.get('NextContinuationToken')


def upload_file(local_path, s3_path, s3 = s3):
    s3.upload_file(local_path, Bucket=S3_BUCKET, Key=s3_path)
    return True

# # def load_s3_yaml(s3, key):

# def load_s3_yaml(key, s3=s3):
#     obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
#     return yaml.safe_load(obj['Body'].read())

# def save_s3_yaml(obj, key, s3=s3):
#     buffer = StringIO()
#     yaml.safe_dump(obj, buffer)
#     s3.put_object(Body=buffer.getvalue(), Bucket=S3_BUCKET, Key=key)
#     return True