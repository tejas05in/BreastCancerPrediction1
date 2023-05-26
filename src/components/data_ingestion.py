from pymongo import MongoClient
from tj_mongo import MongoUrl
import pandas as pd
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
   
  
# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":   
  
  data = get_data()
  data.to_csv("D:/Python\MLProjectsPW/BreastCancerPredicition1/notebooks/data/breast_cancer.csv",index=False)