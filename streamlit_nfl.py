#Use below to run
# streamlit run /Users/SwagMawi/PycharmProjects/streamlit_test/.venv/streamlit_nfl.py

#Let's start out by installing/importing our packages

import streamlit as st
import pandas as pd
import numpy as np
# import nfl_data_py as nfl
import plotly.express as px
import requests
import json
import os
import s3fs

#Before anything else, let's load current data

#Our data lives in S3 so we need to authenticate prior to pulling this data
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

#Now let's pull the data
# @st.cache_data
def load_current_data():
#Now we can pull our data
    data = pd.read_csv(
    "s3://nfl.data/nfl_current_data.csv",
                    storage_options={
                        "key": access_key,
                        "secret": secret_key,
                        }
                        ).reset_index(drop=True)
    # data = data.drop('Unnamed: 0', axis=1)
    data.columns = data.columns.str.lower()
    data['season'] = data['season'].astype('str')
    return data

current_data = load_current_data()

#Let's provide the prediction, then after stats can follow
st.title('Model Prediction')

#Remove the datapoints that the model won't accept
if current_data['week'].max() < 3:
    st.write(f'''
    The model requires 3 weeks of data to make a prediction. 
    Come back right before the first game of week 4 to see our prediction!''')
else:

    # Select your team
    team = st.selectbox('Team', current_data['team'].unique(), placeholder='Select your team', key='model_team')

    #Lets load a photo of your team
    # images = nfl.import_team_desc()
    # image = images[images['team_abbr'] == team]['team_logo_espn']
    # st.image(image)

    #Filter for data from the selected team
    model_data = current_data[current_data['team'] == team]

    #Drop the columns and reduce to the latest week of data
    model_data = model_data.drop(['season', 'week', 'team', 'cover_shifted',
                        'cover_allowed_shifted', 'points_scored_shifted',
                        'scoring_margin_shifted', 'points_allowed_shifted',
                        'scoring_margin_allowed_shifted','total_points_scored_shifted','overtime',
                      'is_cover', 'is_winner', 'is_over'
                      ], axis=1)
    model_data = model_data[-1:]

    #In case there are null values
    model_data = model_data.fillna(0)

    #Save some metrics for the app output
    spread = model_data['spread'].values[0]
    cover_record = model_data['is_cover_record_shifted'].values[0]

    #Lets now get the data ready to feed to our model
    model_data = list(model_data.values[0])
    #The app is written to take a dictionary with 'features' as the key
    #May need to remove that and simplify
    json_data = {'features':model_data}

    #Now we have the app communicate with the model
    # response = requests.post(
    #         "http://localhost:5000/predict",
    #         json=json_data
    #     )

    response = requests.post(
        url="http://98.80.161.146:81/predict",
        json=json_data
    )

    #We then pull in the model prediction
    prediction = response.json()['prediction']

    #Then we write the prediction on the app
    st.write(f'''Based on this team\'s performance, and given the spread is set at {spread},
            the model believes the team has a {prediction*100:.0f}% chance of covering the spread.
            So far this season, this team has covered the spread {cover_record*100:.0f}% of the time''')

################################
####### Current Data #######
################################

#Lets give the page a title
st.title('Current Season Data')

st.subheader('Performance During This Season')

#Now we will filter the data set for this team
data = current_data.copy()
#Take the team the user already selected
data = data[data['team'] == team]

#Now we can add a line chart based on what column the user wants to see

#Select the metric you wish to see
#While I have hundreds of metrics, people usually look at ~10 or so

metric_options = ['points_scored_shifted', 'points_allowed_shifted', 'passing_yards_shifted',
                  'passing_tds_shifted', 'rushing_yards_shifted', 'rushing_tds_shifted',
                  'scoring_margin_shifted', 'scoring_margin_allowed_shifted', 'interceptions_shifted']

# metric_options = [i.replace('_shifted', '').replace('_', ' ').title() for i in metric_options]

st.write('If there are other metrics you\'d like to see, reach out at d_fasil[at]hotmail.com')

metric = st.selectbox('Metric', metric_options,
         placeholder='Select your desired metric', key='current_metric')

#Now lets get the data needed for the line chart
def line_data():
    line_data = data[['week', metric]]
    return line_data

#Now we can plot the chart
if 'rank' in metric:
    fig = px.line(line_data(),
                  x='week',
                  y=metric)
    #This inverses the chart so that the plot rises with a better (lower integer) rank
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
else:
    fig = px.line(line_data(),
                  x='week',
                  y=metric)
    st.plotly_chart(fig, theme="streamlit",use_container_width=True)

################################
####### Historical Data #######
################################

#Lets give the page a title
st.title('Historical Data')

#Our data lives in S3 so we need to authenticate prior to pulling this data
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

#This helps us cache the data once loaded
# @st.cache_data
def load_data():
#Now we can pull our data
    data = pd.read_csv(
    "s3://nfl.data/nfl_historical_data.csv",
                    storage_options={
                        "key": access_key,
                        "secret": secret_key,
                        }
                        ).reset_index(drop=True)
    data.columns = [str(i) for i in data.columns]
    # data = data.drop('Unnamed: 0', axis=1)
    data['season'] = data['season'].astype('str')
    data = data.dropna()
    return data

historical_data = load_data()

st.subheader('Performance Over Time')
st.write('If there are other metrics you\'d like to see, reach out at d_fasil[at]hotmail.com')

#Now we will filter the data set for this team
data = historical_data.copy()
#Take the team the user already selected
data = data[data['team'] == team]

#We'll do the same for season
season = st.selectbox('Season', historical_data['season'].unique(), placeholder='Select season'
                      , key='historical_season')
#Now we will filter the data set for this season
data = data[data['season'] == season]

#Lets limit the stats
metric_options_hist = ['points_scored', 'points_allowed', 'passing_yards',
                  'passing_tds', 'rushing_yards', 'rushing_tds',
                  'scoring_margin', 'scoring_margin_allowed', 'interceptions']


#Select the metric you wish to see
metric = st.selectbox('Metric', metric_options_hist,
         placeholder='Select your desired metric', key='historical_metric')

#Now lets get the data needed for the line chart
def line_data():
    line_data = data[['week', metric]]
    return line_data

#Now we can plot the chart
if 'rank' in metric:
    fig = px.line(line_data(),
                  x='week',
                  y=metric)
    #This inverses the chart so that the plot rises with a better (lower integer) rank
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
else:
    fig = px.line(line_data(),
                  x='week',
                  y=metric)
    st.plotly_chart(fig, theme="streamlit",use_container_width=True)