import pandas as pd
import streamlit as st
import joblib

# model = joblib.load('model_et.joblib')
# ['host_is_superhost' 'host_listings_count' 'latitude' 'longitude'
#  'accommodates' 'bathrooms' 'bedrooms' 'beds' 'extra_people'
#  'minimum_nights' 'instant_bookable' 'is_business_travel_ready' 'ano'
#  'mes' 'n_amenities' 'property_type_Apartment'
#  'property_type_Bed and breakfast' 'property_type_Condominium'
#  'property_type_Guest suite' 'property_type_Guesthouse'
#  'property_type_Hostel' 'property_type_House' 'property_type_Loft'
#  'property_type_Other' 'property_type_Serviced apartment'
#  'room_type_Entire home/apt' 'room_type_Hotel room'
#  'room_type_Private room' 'room_type_Shared room' 'bed_type_Other'
#  'bed_type_Real Bed' 'cancellation_policy_flexible'
#  'cancellation_policy_moderate' 'cancellation_policy_strict'
#  'cancellation_policy_strict_14_with_grace_period']

# ? Separate properties according to its types
x_numeric = {
    'latitude': 0,
    'longitude': 0,
    'accommodates': 0,
    'bathrooms': 0,
    'bedrooms': 0,
    'beds': 0,
    'extra_people': 0,
    'minimum_nights': 0,
    'ano': 0,
    'mes': 0,
    'n_amenities': 0,
    'host_listings_count': 0,
}

x_tf = {'host_is_superhost': 0, 'instant_bookable': 0}

x_lists = {
    'property_type': [
        'Apartment',
        'Bed and breakfast',
        'Condominium',
        'Guest suite',
        'Guesthouse',
        'Hostel',
        'House',
        'Loft',
        'Other',
        'Serviced apartment',
    ],
    'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
    'cancellation_policy': [
        'flexible',
        'moderate',
        'strict',
        'strict_14_with_grace_period',
    ],
}

diction = {}
for item in x_lists:
    for value in x_lists[item]:
        diction[f'{item}_{value}'] = 0
print(diction)
#!Create buttons for streamlit
# numericals
for item in x_numeric:
    if item == 'latitude' or item == 'longitude':
        value = st.number_input(f'{item}', step=0.00001, value=0.0, format="%.5f")
    elif item == 'extra_people':
        value = st.number_input(f'{item}', step=0.01, value=0.0)
    else:
        value = st.number_input(f'{item}', step=1, value=0)
    x_numeric[item] = value

for item in x_tf:
    value = st.selectbox(f'{item}', ('Yes', 'No'))
    if value == 'Yes':
        x_tf[item] = 1
    else:
        x_tf[item] = 0

for item in x_lists:
    value = st.selectbox(f'{item}', x_lists.get(item))
    diction[f'{item}_{value}'] = 1

button = st.button('Predict build value')

# streamlit run deploy_proj_airbnb.py
# 57
# customize our fields

if button:
    diction.update(x_numeric)
    diction.update(x_tf)
    values_x = pd.DataFrame(diction, index=[0])
    model = joblib.load('model_et.joblib')
    price = model.predict(values_x)
    st.write(price)
