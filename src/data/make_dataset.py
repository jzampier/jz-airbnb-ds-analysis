"""Script to create our dataset from csv files"""
#! Import Library and Data Bases
import pandas as pd
import pathlib
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px


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
Analysis of Discret Numerical Features
done - host_listings_count         float64
done - accommodates                  int64
done - bathrooms                   float64
done - bedrooms                    float64
done - beds                        float64
done - guests_included               int64
done - minimum_nights                int64
done - maximum_nights                int64
done - number_of_reviews             int64
"""
# *host_listings_count
diagrama_caixa(base_airbnb.get('host_listings_count'))
graph_bar(base_airbnb.get('host_listings_count'))

# ? We can remove outliers for this category because people with 6 or more
# ? buildings are not our targets to analyze
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'host_listings_count')
print(f'Removed {removed_rows} rows.')

# *accommodates
diagrama_caixa(base_airbnb.get('accommodates'))
graph_bar(base_airbnb.get('accommodates'))

# ? We can remove outliers for this category
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'accommodates')
print(f'Removed {removed_rows} rows.')

# *bathrooms
diagrama_caixa(base_airbnb.get('bathrooms'))
plt.figure(figsize=(15, 5))
sb.barplot(
    x=base_airbnb.get('bathrooms').value_counts().index,
    y=base_airbnb.get('bathrooms').value_counts(),
)

# ? We can remove outliers for this category
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'bathrooms')
print(f'Removed {removed_rows} rows.')

# *bedrooms
diagrama_caixa(base_airbnb.get('bedrooms'))
graph_bar(base_airbnb.get('bedrooms'))

# ? We can remove outliers for this category
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'bedrooms')
print(f'Removed {removed_rows} rows.')

# *beds
diagrama_caixa(base_airbnb.get('beds'))
graph_bar(base_airbnb.get('beds'))

# ? We can remove outliers for this category
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'beds')
print(f'Removed {removed_rows} rows.')

# *guests_included
diagrama_caixa(base_airbnb.get('guests_included'))
graph_bar(base_airbnb.get('guests_included'))

# ? We're going to remove this feature from our analysis due problems related
# ? to fullfilled fields

base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape

# *minimum_nights
diagrama_caixa(base_airbnb.get('minimum_nights'))
graph_bar(base_airbnb.get('minimum_nights'))

# ? We can remove outliers for this category
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'minimum_nights')
print(f'Removed {removed_rows} rows.')

# *maximum_nights
diagrama_caixa(base_airbnb.get('maximum_nights'))
graph_bar(base_airbnb.get('maximum_nights'))

# ? We can remove this feature from our analysis
base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape

# *number_of_reviews
diagrama_caixa(base_airbnb.get('number_of_reviews'))
graph_bar(base_airbnb.get('number_of_reviews'))

# ? We can remove this feature from our analysis
base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape

"""
Analysis of text Features

done - property_type                object
done - room_type                    object
done - bed_type                     object
amenities                    object
done - cancellation_policy          object
"""
# *property_type
tabhouse_types = base_airbnb.get('property_type').value_counts()
print(tabhouse_types)
# ?Using count_plot
plt.figure(figsize=(15, 5))
graph = sb.countplot(x='property_type', data=base_airbnb)
graph.tick_params(axis='x', rotation=90)

# ?Group every kind of property lower than 2000 in 'other' category
# ?Create a list of properties that have lower than 2000 occurrences
columns_to_group = [
    type for type in tabhouse_types.index if tabhouse_types.get(type) < 2000
]
print(columns_to_group)

for ptype in columns_to_group:
    base_airbnb.loc[
        base_airbnb.get('property_type') == ptype, 'property_type'
    ] = 'Other'
print(base_airbnb.get('property_type').value_counts())

# *room_type
print(base_airbnb.get('room_type').value_counts())
# ?Using count_plot
plt.figure(figsize=(15, 5))
graph = sb.countplot(x='room_type', data=base_airbnb)
graph.tick_params(axis='x', rotation=90)

# *bed_type
print(base_airbnb.get('bed_type').value_counts())
# ?Using count_plot
plt.figure(figsize=(15, 5))
graph = sb.countplot(x='bed_type', data=base_airbnb)
graph.tick_params(axis='x', rotation=90)

bed_typ = base_airbnb.get('bed_type').value_counts()
columns_to_group = [type for type in bed_typ.index if bed_typ.get(type) < 10_000]
print(columns_to_group)

for type in columns_to_group:
    base_airbnb.loc[base_airbnb.get('bed_type') == type, 'bed_type'] = 'Other'
print(base_airbnb.get('bed_type').value_counts())

plt.figure(figsize=(15, 5))
graph = sb.countplot(x='bed_type', data=base_airbnb)
graph.tick_params(axis='x', rotation=90)

# *cancellation_policy
print(base_airbnb.get('cancellation_policy').value_counts())
# ?Using count_plot
plt.figure(figsize=(15, 5))
graph = sb.countplot(x='cancellation_policy', data=base_airbnb)
graph.tick_params(axis='x', rotation=90)

canc_pol = base_airbnb.get('cancellation_policy').value_counts()
print(canc_pol.index)
print(canc_pol.values)

columns_to_group = [type for type in canc_pol.index if canc_pol.get(type) < 10_000]
print(columns_to_group)

for type in columns_to_group:
    base_airbnb.loc[
        base_airbnb.get('cancellation_policy') == type, 'cancellation_policy'
    ] = 'strict'
print(base_airbnb.get('cancellation_policy').value_counts())

plt.figure(figsize=(15, 5))
graph = sb.countplot(x='cancellation_policy', data=base_airbnb)
graph.tick_params(axis='x', rotation=90)

# ***amenities***
# ? It's almost impossible to compare each category of amenities, so we're going
# ? to count how many amenities are in each house/apartment (more amenities, more expensive)
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
# Drop the former category 'amenities'
base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape
diagrama_caixa(base_airbnb.get('n_amenities'))
graph_bar(base_airbnb.get('n_amenities'))

# ? We can remove outliers for this category
base_airbnb, removed_rows = rm_outliers(base_airbnb, 'n_amenities')
print(f'Removed {removed_rows} rows.')

# *Visualization of maps
map_sample = base_airbnb.sample(n=100000)
map_center = {
    'lat': map_sample.get('latitude').mean(),
    'lon': map_sample.get('longitude').mean(),
}
map_la_lo_price = px.density_mapbox(
    map_sample,
    lat='latitude',
    lon='longitude',
    z='price',
    radius=2.5,
    center=map_center,
    zoom=10,
    mapbox_style='stamen-terrain',
)
map_la_lo_price

#! Encoding
"""
Transform all text information in numbers
Features with values True or False we're going to replace to 1 or 0
Categorical features we're going to use Dummies to encode them.
"""
# Print features list
print(base_airbnb.columns)
tf_columns = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
categorical_columns = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']

# Create an encoded copy of our dataframe
encoded_base_airbnb = base_airbnb.copy()
for column in tf_columns:
    encoded_base_airbnb.loc[encoded_base_airbnb.get(column) == 't', column] = 1
    encoded_base_airbnb.loc[encoded_base_airbnb.get(column) == 'f', column] = 0
# Print 1 row to check if its ok
print(encoded_base_airbnb.iloc[0])

# Using Dummies to adjust Categorical values using pandas pd.get_dummies()
encoded_base_airbnb = pd.get_dummies(
    data=encoded_base_airbnb, columns=categorical_columns
)
encoded_base_airbnb.shape

#! Done with data adjustments, ready to modelling
