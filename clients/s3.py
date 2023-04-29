import os
import boto3
from clients import ENV
import constants
from typing import BinaryIO


def upload(
    serialized_data: bytes,
    s3_path: str,
    env: str,
    bucket_name: str = constants.S3_BUCKET_NAME,
) -> None:
    """
    Upload serialized data to an S3 bucket at a specified path. The file is to be
        uploaded from the memory directly, without saving it to disk.

    Parameters
    ----------
    serialized_data: Any
        Serialized data (e.g. model, table) to upload to S3. The object to upload has
        to be converted to byte stream first, examples below.
    s3_path: str
        Path within the S3 bucket to store the data.
    env: str
        Environment. Can be either `dev` or `prod`.
    bucket_name: str
        Name of the S3 bucket to upload the file to.

    Returns
    -------
    None

    Examples
    --------
    To upload an object as pickle file, serialize it with `dumps` as shown below.

        estimator = ...
        upload(
            serialized_data=pickle.dumps(estimator.model),
            s3_path="models/date=2023-01-01/model.pkl",
            env="dev",
        )
        --------
        To upload a Parquet file, use pandas `to_parquet` method.
        df = ...
        upload(
            serialized_data=pd.DataFrame.to_parquet(df, index=False),
            s3_path="data/date=2023-01-01/data.parquet",
            env="dev",
        )

    """
    _validate_env(env)
    session = _get_session()
    s3 = session.client("s3")
    s3.put_object(
        Body=serialized_data,
        Bucket=bucket_name,
        Key=os.path.join(env, s3_path),
        ACL="bucket-owner-full-control",
    )


def download(
    s3_path: str,
    env: str,
    bucket_name: str = constants.S3_BUCKET_NAME,
) -> BinaryIO:
    """
    Download a file from an S3 bucket to byte stream in memory. The downloaded file
        has to be deserialized (examples below).
        
    Parameters
    ----------
    s3_path: str
        Path within the S3 bucket to store the data.
    env: str
        Environment. Can be either `dev` or `prod`.
    bucket_name: str
        Name of the S3 bucket to upload the file to.
    Returns
    -------
    Data as bytes stream.

    Examples
    --------
    To download and read a pickle file, deserialize it with `loads` as shown below.

        import pickle
        bytes_data = download(s3_path="models/ECA/2023-01-01.pkl", env="dev")
        estimator.model = pickle.loads(bytes_data)

    To read a CSV file, use pandas `read_csv` method as below.

        import io
        bytes_data = download(s3_path="predictions/ECA/2023-01-01.csv", env="dev")
        df = pd.read_csv(io.StringIO(bytes_data.decode("utf-8")))

    """
    _validate_env(env)
    session = _get_session()
    s3 = session.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=os.path.join(env, s3_path))
    return response["Body"].read()


def _validate_env(env: str) -> None:
    if env not in ("prod", "dev"):
        raise ValueError(f"Invalid env variable '{env}'. Expected 'prod' or 'dev'.")


def _get_session():
    return boto3.Session(
        aws_access_key_id=ENV["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=ENV["AWS_SECRET_ACCESS_KEY"],
    )
