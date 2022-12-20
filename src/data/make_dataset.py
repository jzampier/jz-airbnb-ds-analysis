"""Script to create our dataset from csv files"""
#! Import Library and Data Bases
import pandas as pd
import pathlib
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


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

    df: pd.DataFrame = pd.read_csv(path_db / file.name)
    df['ano'] = year
    df['mes'] = month

    base_airbnb: pd.DataFrame = pd.concat([base_airbnb, df], axis=0)
    print(file.name)

#! Data treatment
# * Exclude unwanted columns
# ? As we have lots of unwanted columns, an new tabular file should be created
# ? in order to allow a better column analysis and let us to chose which column we
# ? are going to drop of our dataset. This file should have the 1st 1000 records
# ? Column types we're going to drop:
# ? 1. Id's, Links, non relevant information
# ? 2. Repeated or similar columns E.g (Date , Year, Month)
# ? 3. Free text columns
# ? 4. Columns where almost every value is equal

# print(list(base_airbnb.columns))
# base_airbnb.head(1000).to_csv('../../data/processed/first_1000records.csv')

# # *Auxiliar code to check integrity of some of columns
# print(base_airbnb.get('experiences_offered').value_counts())
# print(
#     (
#         base_airbnb.get('host_listings_count')
#         == base_airbnb.get('host_total_listings_count')
#     ).value_counts()
# )
# print(base_airbnb.get('square_feet').isnull().sum())
# * After the column analysis, we decided to keep these columns
colunas: list = [
    'host_response_time',
    'host_response_rate',
    'host_is_superhost',
    'host_listings_count',
    'latitude',
    'longitude',
    'property_type',
    'room_type',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'bed_type',
    'amenities',
    'price',
    'security_deposit',
    'cleaning_fee',
    'guests_included',
    'extra_people',
    'minimum_nights',
    'maximum_nights',
    'number_of_reviews',
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'instant_bookable',
    'is_business_travel_ready',
    'cancellation_policy',
    'ano',
    'mes',
]

base_airbnb: pd.DataFrame = base_airbnb.loc[:, colunas]

# * Adjust Nan values
# ? Identify which columns have NaN values and the proportion of this values
# ? Noted that we had lots of NaN values. Initially we gonna drop columns with more
# ? than 300k values
for coluna in base_airbnb:
    if base_airbnb.get(coluna).isnull().sum() > 300000:
        base_airbnb: pd.DataFrame = base_airbnb.drop(coluna, axis=1)

print(base_airbnb.isnull().sum())
base_airbnb: pd.DataFrame = base_airbnb.dropna()
print(base_airbnb.shape)

# * Verify data types for each column
print(base_airbnb.dtypes)
print('-' * 60)
print(base_airbnb.iloc[0])

# * Price and extra_people are being recognized as objects, we have to change
# * them to float
# todo-> Price, extra_people
base_airbnb['price'] = base_airbnb.get('price').str.replace('$', '')
base_airbnb['price'] = base_airbnb.get('price').str.replace(',', '')
base_airbnb['price'] = base_airbnb.get('price').astype(np.float32, copy=False)

base_airbnb['extra_people'] = base_airbnb.get('extra_people').str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb.get('extra_people').str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb.get('extra_people').astype(
    np.float32, copy=False
)
print(base_airbnb.dtypes)

"""
! Exploratory analysis treat Outliers
-We're going analyse every feature and compare them for:
1-Find correlationship between features and decide if we gonna keep both
features
2-Remove outliers
3-Check if all features that we have, make sense to our model. If some of them
don't help us, we have to remove

-We gonna start wiith price and extra_people. These are the continuous numeric
values.

-After that, we are going to analyse columns with discrete values.

-Finally, we gonna evaluate text columns and which ones make sense to our
analysis
"""
# ? Creating our plt figure
plt.figure(figsize=(15, 10))
# ? Pandas function for correlationship
# print(base_airbnb.corr())
# ? Using seaborn to plot the correlationship between columns
sb.heatmap(base_airbnb.corr(), annot=True, cmap='Greens')

# ? Exclude outliers
"""
Defining some functions to analyse and remove outliers
Amplitude = Q3 - Q1
Lower limit = Q1 -1.5 x Amplitude
Upper limit = Q3 + 1,5 x Amplitude
"""


def limites(column: list):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude


def rm_outliers(df: pd.DataFrame, col_name: str) -> tuple[pd.DataFrame, int]:
    """remove outliers from column and count how many rows were excluded

    Keyword arguments:
    df -- dataframe
    col_name -- column name
    Return: Exclude outliers and returns a dataframe withou these outliers
    """
    quant_rows = df.shape[0]
    lower_lim, upper_lim = limites(df.get(col_name))
    df = df.loc[((df.get(col_name) >= lower_lim) & (df.get(col_name) <= upper_lim)), :]
    act_rows_quant = quant_rows - df.shape[0]
    return df, act_rows_quant


# print(limites(base_airbnb.get('price')))

"""
Box plot
Let's define a function to plot
"""


def diagrama_caixa(column: list):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sb.boxplot(x=column, ax=ax1)
    ax2.set_xlim(limites(column))
    sb.boxplot(x=column, ax=ax2)


def histogram(column: list):
    plt.figure(figsize=(15, 5))
    # sb.histplot(column)
    sb.histplot(column, kde=True)
    # sb.histplot(column, kde=True, stat="density", kde_kws=dict(cut=3))


def graph_bar(column: list):
    plt.figure(figsize=(15, 5))
    ax = sb.barplot(x=column.value_counts().index, y=column.value_counts())
    ax.set_xlim(limites(column))


diagrama_caixa(base_airbnb.get('price'))
histogram(base_airbnb.get('price'))
# *price
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'price')
print(f'Removed {removed_rows} rows.')
histogram(base_airbnb.get('price'))

# *extra_people
diagrama_caixa(base_airbnb.get('extra_people'))
histogram(base_airbnb.get('extra_people'))

base_airbnb, removed_rows = rm_outliers(base_airbnb, 'extra_people')
print(f'Removed {removed_rows} rows.')
histogram(base_airbnb.get('extra_people'))

"""
done - host_listings_count         float64
accommodates                  int64
bathrooms                   float64
bedrooms                    float64
beds                        float64
guests_included               int64
minimum_nights                int64
maximum_nights                int64
number_of_reviews             int64
"""
# *host_listings_count
diagrama_caixa(base_airbnb.get('host_listings_count'))
graph_bar(base_airbnb.get('host_listings_count'))

# ? We can remove outliers for this category because people with 6 or more
# ? buildings are not our targets to analyze
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'host_listings_count')
print(f'Removed {removed_rows} rows.')


#! Encoding
