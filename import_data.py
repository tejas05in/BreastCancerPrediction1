from database_connect import mongo_operation as mongo
from tj_mongo import MongoUrl
from sklearn.datasets import load_breast_cancer
import pandas as pd

client_url= MongoUrl.client_url
database_name = "demoDB"


def upload_files_to_mongodb(
    mongo_client_con_string,
    database_name):

    mongo_connection = mongo(
        client_url = mongo_client_con_string,
        database_name= database_name,
        collection_name= 'breast_cancer_prediction'
    )

    breast_cancer = load_breast_cancer(as_frame=True)
    df = pd.concat([breast_cancer.data,breast_cancer.target],axis=1)

    mongo_connection.bulk_insert(df)
    print("breast_cancer_prediction dataset is uploaded to mongodb")

upload_files_to_mongodb(
        mongo_client_con_string= client_url,
        database_name = database_name)