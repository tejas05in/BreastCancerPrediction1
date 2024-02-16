import sys
import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from tj_mongo import MongoUrl
import pandas as pd
from dataclasses import dataclass

def get_database():
 
   # Provide the mongodb atlas url to connect python to mongodb using pymongo
   CONNECTION_STRING = MongoUrl.client_url 
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(CONNECTION_STRING)
 
   # Create the database for our example (we will use the same database throughout the tutorial
   return client['demoDB']

def get_data():
   # Get the database
   dbname = get_database()
   collection = dbname['breast_cancer_prediction']
   cursor = collection.find()
   data = []
   for doc in cursor:
       data.append(doc)
   
   df = pd.DataFrame.from_records(data).drop('_id',axis=1)
   return df

@dataclass
class DataIngestionconfig:
   train_data_path:str = os.path.join('artifacts' , 'train.csv')
   test_data_path:str = os.path.join('artifacts' , 'test.csv')
   raw_data_path:str = os.path.join('artifacts','raw.csv')

### create the data ingestion class
class DataIngestion:
   def __init__(self) -> None:
      self.ingestion_config = DataIngestionconfig()

   def initiate_data_ingestion(self):
      logging.info('Data Ingestion method starts')
      try:
         df = pd.read_csv(os.path.join('notebooks/data','breast_cancer.csv'))
         # df = get_data()
         logging.info("Dataset read as pandas Dataframe")

         os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
         df.to_csv(self.ingestion_config.raw_data_path,index=False)

         logging.info('Raw data is created')

         train_set , test_set = train_test_split(df,test_size=0.3 , random_state=42)
         train_set.to_csv(self.ingestion_config.train_data_path,index =False,header = True)
         test_set.to_csv(self.ingestion_config.test_data_path,index =False,header = True)
         logging.info('Ingestion of data is completed')

         return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
         )

      except Exception as e:
         logging.info('Exception occured at data ingestion stage')
         raise CustomException(e,sys)
      

if __name__ == "__main__":   
  
  data = get_data()
  file_name = 'breast_cancer.csv'
  data.to_csv(os.path.join('notebooks/data',file_name),index=False)