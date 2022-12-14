"""Script to create our dataset from csv files"""
#! Import Library and Data Bases
import pandas as pd
import pathlib

months: dict = {
    'jan': 1,
    'fev': 2,
    'mar': 3,
    'abr': 4,
    'mai': 5,
    'jun': 6,
    'jul': 7,
    'ago': 8,
    'set': 9,
    'out': 10,
    'nov': 11,
    'dez': 12,
}
path_db: str = pathlib.Path('../../data/raw')
base_airbnb: pd.DataFrame = pd.DataFrame()


for file in path_db.iterdir():

    month_name: str = file.name[:3]
    month: str = months.get(month_name)

    year: str = file.name[-8:]
    year: int = int(year.replace('.csv', ''))

    df = pd.read_csv(path_db / file.name)
    df['ano'] = year
    df['mes'] = month

    base_airbnb = pd.concat([base_airbnb, df], axis=0)
    print(file.name)

#! Consolidate the dataset
base_airbnb

#! Exclude unwanted columns
# *9

#! Adjust Nan values

#! Verify data types for each column

#! Exploratory analysis treat Outliers

#! Encoding
