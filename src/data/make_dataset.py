"""Script to create our dataset from csv files"""
#! Import Library and Data Bases
import pandas as pd
import pathlib


path_db = pathlib.Path('../../data/raw')
base_airbnb = pd.DataFrame()


for file in path_db.iterdir():
    print(file.name)

abril2018_df = pd.read_csv(r'../../data/raw/abril2018.csv')
#! Consolidate the dataset

#! Exclude unwanted columns

#! Adjust Nan values

#! Verify data types for each column

#! Exploratory analysis treat Outliers

#! Encoding
