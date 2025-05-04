#Lets import our packages

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests as r
from http.client import IncompleteRead
import nfl_data_py as nfl
import boto3
import os

#We'll get actual performance data from nfl data py (Python vesion of nflfastR) 
#Github repo: https://github.com/cooperdff/nfl_data_py


if datetime.today().month >= 9:
    current_year = datetime.today().year
else:
    current_year = datetime.today().year - 1

years = [current_year]


weekly_data = nfl.import_weekly_data(years)

#Let's filter for only regular season
weekly_data = weekly_data[weekly_data['season_type'] == 'REG'].reset_index(drop=True)


#I want to organize these columns based on the team that's listed and their opponent
#For the opponent, I want to capture these columns as defensive stats (yards allowed, etc)
#The pfr table has opponent as a column, so let's use that

weekly_pfr = nfl.import_weekly_pfr('pass',years)

weekly_pfr = weekly_pfr[weekly_pfr['game_type'] == 'REG'].reset_index(drop=True)

team_opp = weekly_pfr[['season', 'week', 'team', 'opponent']]

#Due to raiders moving, let's keep all raider rows as LV
team_opp.loc[team_opp['team'] == 'OAK', 'team'] = 'LV'
team_opp.loc[team_opp['opponent'] == 'OAK', 'opponent'] = 'LV'


#Table is based on passing, so sometimes there's two QBs, so we need to drop duplicates as we just want opponent
team_opp = team_opp.drop_duplicates()

#Lets add an opponent column to the 
weekly_data = weekly_data.merge(team_opp, left_on = ['recent_team', 'season', 'week'],
                                right_on = ['team', 'season', 'week'],
                                how='left'
                               )


#Definitions for columns can be found here: https://www.nflfastr.com/reference/calculate_player_stats.html
#We can group these by team by week to get either the sum or avg of certain columns
#Group by team by week b/c this is at the player level

weekly_agg = weekly_data.groupby(['season', 'week', 'recent_team']).agg({'completions':'sum',
                                                              'attempts':'sum',
                                                              'passing_yards':'sum' ,
                                                             'passing_tds':'sum', 
                                                             'interceptions':'sum',
                                                             'sacks':'sum',
                                                             'sack_yards':'sum',
                                                             'sack_fumbles':'sum',
                                                             'sack_fumbles_lost':'sum',
                                                             'passing_air_yards':'sum',
                                                             'passing_yards_after_catch':'sum',
                                                             'passing_first_downs':'sum',
                                                             'passing_epa':['mean','sum'],
                                                             'passing_2pt_conversions':'sum',
                                                             'dakota':'mean',
                                                             'carries':'sum',
                                                             'rushing_yards':'sum',
                                                             'rushing_tds':'sum',
                                                             'rushing_fumbles':'sum',
                                                             'rushing_fumbles_lost':'sum',
                                                             'rushing_first_downs':'sum',
                                                             'rushing_epa':['mean','sum'],
                                                             'rushing_2pt_conversions':'sum',
                                                             'receptions':'sum',
                                                             'targets':'sum',
                                                             'receiving_yards':'sum',
                                                             'receiving_tds':'sum',
                                                             'receiving_fumbles':'sum',
                                                             'receiving_fumbles_lost':'sum',
                                                             'receiving_air_yards':'sum',
                                                             'receiving_yards_after_catch':'sum',
                                                             'receiving_first_downs':'sum',
                                                             'receiving_epa':['mean','sum'],
                                                             'receiving_2pt_conversions':'sum',
                                                             'wopr':['mean','sum'],
                                                             'special_teams_tds':'sum',
                                                             'fantasy_points':'sum',
                                                             'fantasy_points_ppr':'sum'
                                                              }).reset_index()


weekly_agg = weekly_agg.droplevel(1,axis=1)

#Columns to re-create with game totals:
#PACR: Passing yards / Passing air yards
#RACR: Receiving yards / Receiving air yards

weekly_agg['pacr'] = weekly_agg['passing_yards'] / weekly_agg['passing_air_yards']
weekly_agg['racr'] = weekly_agg['receiving_yards'] / weekly_agg['receiving_air_yards']

#Since we dropped the multi-level index, some columns have the same name b/c it's referring mean or sum
#Manually renaming them because identifying by name will change all columns with the same name

weekly_agg.columns = ['season','week', 'team', 'completions', 'attempts', 'passing_yards',
       'passing_tds', 'interceptions', 'sacks', 'sack_yards', 'sack_fumbles',
       'sack_fumbles_lost', 'passing_air_yards', 'passing_yards_after_catch',
       'passing_first_downs', 'avg_passing_epa', 'total_passing_epa', 'passing_2pt_conversions', 
        'dakota', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
       'rushing_first_downs', 'avg_rushing_epa', 'total_rushing_epa','rushing_2pt_conversions',
        'receptions', 'targets', 'receiving_yards',
       'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost',
       'receiving_air_yards', 'receiving_yards_after_catch',
       'receiving_first_downs', 'avg_receiving_epa', 'total_receiving_epa',
       'receiving_2pt_conversions', 'avg_wopr', 'total_wopr', 'special_teams_tds',
       'fantasy_points', 'fantasy_points_ppr', 'pacr', 'racr']


#This dataset doesn't have defensive stats
#But if we group by opponent, we can get the same stats but as "allowed"
#For example, bucs v dal week 1 2021, bucs passing yards = dal passing yards allowed

weekly_agg_def = weekly_data.groupby(['season', 'week', 'opponent']).agg({'completions':'sum',
                                                              'attempts':'sum',
                                                              'passing_yards':'sum' ,
                                                             'passing_tds':'sum', 
                                                             'interceptions':'sum',
                                                             'sacks':'sum',
                                                             'sack_yards':'sum',
                                                             'sack_fumbles':'sum',
                                                             'sack_fumbles_lost':'sum',
                                                             'passing_air_yards':'sum',
                                                             'passing_yards_after_catch':'sum',
                                                             'passing_first_downs':'sum',
                                                             'passing_epa':['mean','sum'],
                                                             'passing_2pt_conversions':'sum',
                                                             'dakota':'mean',
                                                             'carries':'sum',
                                                             'rushing_yards':'sum',
                                                             'rushing_tds':'sum',
                                                             'rushing_fumbles':'sum',
                                                             'rushing_fumbles_lost':'sum',
                                                             'rushing_first_downs':'sum',
                                                             'rushing_epa':['mean','sum'],
                                                             'rushing_2pt_conversions':'sum',
                                                             'receptions':'sum',
                                                             'targets':'sum',
                                                             'receiving_yards':'sum',
                                                             'receiving_tds':'sum',
                                                             'receiving_fumbles':'sum',
                                                             'receiving_fumbles_lost':'sum',
                                                             'receiving_air_yards':'sum',
                                                             'receiving_yards_after_catch':'sum',
                                                             'receiving_first_downs':'sum',
                                                             'receiving_epa':['mean','sum'],
                                                             'receiving_2pt_conversions':'sum',
                                                             'wopr':['mean','sum'],
                                                             'special_teams_tds':'sum',
                                                             'fantasy_points':'sum',
                                                             'fantasy_points_ppr':'sum'
                                                              }).reset_index()

weekly_agg_def = weekly_agg_def.droplevel(1,axis=1)


weekly_agg_def['pacr'] = weekly_agg_def['passing_yards'] / weekly_agg_def['passing_air_yards']
weekly_agg_def['racr'] = weekly_agg_def['receiving_yards'] / weekly_agg_def['receiving_air_yards']

#Since this is defensive allowed, we can take the columns from weekly_agg and add _allowed
weekly_agg_def.columns = [i+'_allowed' if i not in ['season', 'week', 'team', 'opponent'] else i for i in weekly_agg.columns]

#Sort the table by year, team, and week
weekly_agg = weekly_agg.sort_values(by=['season','team','week']).reset_index(drop=True)
weekly_agg_def = weekly_agg_def.sort_values(by=['season','team','week']).reset_index(drop=True)

#Now let's join the two tables so we have both offense and defensive stats
weekly_agg = weekly_agg.merge(weekly_agg_def, on=['season','team','week'])

#Lets get rolling values for these columns
original_week = weekly_agg['week'].copy()


weekly_agg_rolling = weekly_agg.groupby(['season', 'team'])[
    [i for i in weekly_agg.columns if i not in ['season', 'week', 'team']]
].rolling(3).mean().reset_index()


weekly_agg_rolling = weekly_agg_rolling.drop('level_2', axis=1)
weekly_agg_rolling['week'] = original_week


weekly_agg_rolling.columns = [i+'_rolling' if i not in ['season', 'week', 'team', 'opponent'] \
                              else i for i in weekly_agg_rolling.columns]

#Lets join the two tables
weekly_agg = weekly_agg.merge(weekly_agg_rolling, on=['season', 'week', 'team'])

#Lets get a rank of all our metrics
columns_to_rank = [i for i in weekly_agg.columns if i not in ['season', 'week', 'team', 'opponent']]

for col in columns_to_rank:
    rank_col_name = f'rank_{col}_weekly'
    weekly_agg[rank_col_name] = weekly_agg.groupby(['season', 'week'])[col].rank(ascending=False)


#Let's rank columns based on what was done earlier in season
#First we get a sum or avg of these columns by season up to that point in the week

#First we'll get the columns that should be summed up (non-avg columns)
columns_to_sum = [i for i in weekly_agg.columns \
                  if i not in ['season', 'week', 'team','avg_passing_epa', 'dakota', 'avg_rushing_epa', 
                  'avg_receiving_epa', 'avg_wopr', 'pacr', 'racr',
                              'avg_passing_epa_allowed', 'dakota_allowed', 'avg_rushing_epa_allowed', 
                  'avg_receiving_epa_allowed', 'avg_wopr_allowed', 'pacr_allowed', 'racr_allowed']
                 and '_rolling' not in i
                 and 'rank' not in i]

for col in columns_to_sum:
    expand_col_name = f'expand_{col}'
    weekly_agg[expand_col_name] = weekly_agg.sort_values(by='week').groupby(['season', 'team'])[col].cumsum()

#Lets check this is working

#Great, lets do this for averages
colums_to_expand_avg = ['avg_passing_epa',
                       'dakota',
                       'avg_rushing_epa',
                       'avg_receiving_epa',
                       'avg_wopr',
                       'pacr', 
                       'racr',
                       'avg_passing_epa_allowed', 'dakota_allowed', 'avg_rushing_epa_allowed', 
                  'avg_receiving_epa_allowed', 'avg_wopr_allowed', 'pacr_allowed', 'racr_allowed']

for col in colums_to_expand_avg:
    expand_col_name = f'expand_{col}'
    cumulative_sum = weekly_agg.sort_values(by='week').groupby(['season', 'team'])[col].cumsum()
    cumulative_count = weekly_agg.sort_values(by='week').groupby(['season', 'team'])[col].cumcount() + 1
    weekly_agg[expand_col_name] = cumulative_sum / cumulative_count


#Now we'll get the ranking for these "expanded" columns
for col in weekly_agg.columns:
    if 'expand_' in col:
        rank_col_name = f'rank_{col}_weekly'
        weekly_agg[rank_col_name] = weekly_agg.groupby(['season', 'week'])[col].rank(ascending=False)


# # Passing Data

weekly_pfr = nfl.import_weekly_pfr('pass',years)
weekly_pfr = weekly_pfr[weekly_pfr['game_type'] == 'REG'].reset_index(drop=True)

#Example/Definitions available here:
#https://www.pro-football-reference.com/players/M/MahoPa00/gamelog/2022/advanced/#all_advanced_passing
#Hover over column names to get definitions


#Noticed that the columns like def_times_blitzed were null, are they always null?
#Also noticed there's no receiving drops

#We'll just drop them

weekly_pfr = weekly_pfr.drop(['def_times_hurried', 
                              'def_times_hitqb', 
                             'def_times_blitzed',
                              'receiving_drop',
                                'receiving_drop_pct'], axis=1)


#Changing raiders to LV b/c of their move
weekly_pfr.loc[weekly_pfr['team'] == 'OAK', 'team'] = 'LV'
weekly_pfr.loc[weekly_pfr['opponent'] == 'OAK', 'opponent'] = 'LV'

#Lets aggregate across week and season
pfr_agg = weekly_pfr.groupby(['season', 'week', 'team']).agg({'passing_drops':'sum',
                                                              'passing_drop_pct':'mean', 
                                                              'passing_bad_throws':'sum', 
                                                              'passing_bad_throw_pct':'mean',
                                                              'times_sacked':'sum',
                                                              'times_blitzed':'sum', 
                                                              'times_hurried':'sum',
                                                              'times_hit':'sum',
                                                              'times_pressured':'sum',
                                                              'times_pressured_pct':'mean'}).reset_index()



#Like before, let's group by opponent to get "allowed" columns
#We'll use this as defensive stats
pfr_agg_def = weekly_pfr.groupby(['season', 'week', 'opponent']).agg({'passing_drops':'sum',
                                                              'passing_drop_pct':'mean', 
                                                              'passing_bad_throws':'sum', 
                                                              'passing_bad_throw_pct':'mean',
                                                              'times_sacked':'sum',
                                                              'times_blitzed':'sum', 
                                                              'times_hurried':'sum',
                                                              'times_hit':'sum',
                                                              'times_pressured':'sum',
                                                              'times_pressured_pct':'mean'}).reset_index()


pfr_agg_def.columns = [i+'_allowed' if i not in ['season', 'team', 'week'] else i for i in pfr_agg.columns ]

#Lets join our offensive and defensive stats
pfr_agg = pfr_agg.sort_values(by=['season','team','week']).reset_index(drop=True)
pfr_agg_def = pfr_agg_def.sort_values(by=['season','team','week']).reset_index(drop=True)


pfr_agg = pfr_agg.merge(pfr_agg_def, on=['season','team','week'])


#Now lets get rolling columns
pfr_agg = pfr_agg.sort_values(by=['season','team','week']).reset_index(drop=True)
original_week = pfr_agg['week'].copy()
pfr_agg_rolling = pfr_agg.groupby(['season', 'team'])[['passing_drops', 'passing_drop_pct',
                                                       'passing_bad_throws',
                                                       'passing_bad_throw_pct', 
                                                       'times_sacked', 
                                                       'times_blitzed',
       'times_hurried', 'times_hit', 'times_pressured', 'times_pressured_pct']].rolling(3).mean().reset_index()

pfr_agg_rolling = pfr_agg_rolling.drop('level_2', axis=1)

pfr_agg_rolling['week'] = original_week

pfr_agg_rolling.columns = [i+'_rolling' if i not in ['season', 'week', 'team', 'opponent'] \
                              else i for i in pfr_agg_rolling.columns]

#Joining rolling to regular table
pfr_agg = pfr_agg.merge(pfr_agg_rolling, on=['season', 'team', 'week'])

#Lets get the ranks for each week for these metrics
columns_to_rank = [i for i in pfr_agg.columns if i not in ['season', 'week', 'team']]

for col in columns_to_rank:
    rank_col_name = f'rank_{col}_weekly'
    pfr_agg[rank_col_name] = pfr_agg.groupby(['season', 'week'])[col].rank(ascending=False)

#Lets get sums/avg up to each part of the season
colums_to_expand_sum = ['passing_drops', 
       'passing_bad_throws', 'times_sacked',
       'times_blitzed', 'times_hurried', 
                        'times_hit', 'times_pressured',
        'passing_drops_allowed', 
       'passing_bad_throws_allowed', 'times_sacked_allowed',
       'times_blitzed_allowed', 'times_hurried_allowed', 
                        'times_hit_allowed', 'times_pressured_allowed']


for col in colums_to_expand_sum:
    expand_col_name = f'expand_{col}'
    pfr_agg[expand_col_name] = pfr_agg.sort_values(by='week').groupby(['season', 'team'])[col].cumsum()

#Lets check this is working

#Great, lets do this for averages
colums_to_expand_avg = ['passing_drop_pct','passing_bad_throw_pct','times_pressured_pct',
                       'passing_drop_pct_allowed','passing_bad_throw_pct_allowed','times_pressured_pct_allowed']

for col in colums_to_expand_avg:
    expand_col_name = f'expand_{col}'
    cumulative_sum = pfr_agg.sort_values(by='week').groupby(['season', 'team'])[col].cumsum()
    cumulative_count = pfr_agg.sort_values(by='week').groupby(['season', 'team'])[col].cumcount() + 1
    pfr_agg[expand_col_name] = cumulative_sum / cumulative_count


#Now we'll get the rank of the expanding columns
for col in pfr_agg.columns:
    if 'expand_' in col:
        rank_col_name = f'rank_{col}_weekly'
        pfr_agg[rank_col_name] = pfr_agg.groupby(['season', 'week'])[col].rank(ascending=False)


weekly_agg = weekly_agg.merge(pfr_agg, on=['season', 'team', 'week'])


# # Rush data

weekly_pfr_rush = nfl.import_weekly_pfr('rush',years)
weekly_pfr_rush = weekly_pfr_rush[weekly_pfr_rush['game_type'] == 'REG'].reset_index(drop=True)

#I see null for receiving_broken_tackles, is this the case for all teams?
#We don't need that column
weekly_pfr_rush =  weekly_pfr_rush.drop('receiving_broken_tackles', axis=1)


weekly_pfr_rush.loc[weekly_pfr_rush['team'] == 'OAK', 'team'] = 'LV'
weekly_pfr_rush.loc[weekly_pfr_rush['opponent'] == 'OAK', 'opponent'] = 'LV'


#Lets get our aggreate values for rushing
#The avg columns we can calculate once aggregated

pfr_agg_rush = weekly_pfr_rush.groupby(['season', 'week', 'team'])[['carries',
                                                                   'rushing_yards_before_contact',
                                                                   'rushing_yards_after_contact',
                                                                   'rushing_broken_tackles']].sum().reset_index()

#I checked PFR to confirm that rushing_yards_before_contact + rushing_yards_after_contact = total rush yards
#Lets get the avg yards before and after contact
#Let's get "allowed columns"
pfr_agg_rush_def = weekly_pfr_rush.groupby(['season', 'week', 'opponent'])[['carries',
                                                                   'rushing_yards_before_contact',
                                                                   'rushing_yards_after_contact',
                                                                   'rushing_broken_tackles']].sum().reset_index()

pfr_agg_rush_def.columns = [i+'_allowed' if i not in ['season', 'team','week', 'opponent'] \
                            else i for i in pfr_agg_rush.columns]

#Lets join the rushing data with the rushing allowed data
pfr_agg_rush = pfr_agg_rush.sort_values(by=['season','team','week']).reset_index(drop=True)
pfr_agg_rush_def = pfr_agg_rush_def.sort_values(by=['season','team','week']).reset_index(drop=True)


pfr_agg_rush = pfr_agg_rush.merge(pfr_agg_rush_def, on=['season', 'week', 'team'])

#Lets get the avg yards before and after contact
pfr_agg_rush['rushing_yards_before_contact_avg'] = pfr_agg_rush['rushing_yards_before_contact'] / pfr_agg_rush['carries']

pfr_agg_rush['rushing_yards_after_contact_avg'] = pfr_agg_rush['rushing_yards_after_contact'] / pfr_agg_rush['carries']

pfr_agg_rush['rushing_yards_before_contact_avg_allowed'] = pfr_agg_rush['rushing_yards_before_contact_allowed'] / pfr_agg_rush['carries_allowed']

pfr_agg_rush['rushing_yards_after_contact_avg_allowed'] = pfr_agg_rush['rushing_yards_after_contact_allowed'] / pfr_agg_rush['carries_allowed']


#Let's get our rolling averages
original_week = pfr_agg_rush['week'].copy()
columns_to_roll = [i for i in pfr_agg_rush.columns if i not in ['season', 'week', 'team']]

pfr_rush_rolling = pfr_agg_rush.groupby(['season', 'team'])[columns_to_roll].rolling(3).mean().reset_index()
pfr_rush_rolling = pfr_rush_rolling.drop('level_2', axis=1)

pfr_rush_rolling['week'] = original_week

pfr_rush_rolling.columns = [i+'_rolling' if i not in ['season', 'team', 'week']\
                            else i for i in pfr_rush_rolling.columns]


#Lets join rolling values to our rush dataset
pfr_agg_rush = pfr_agg_rush.merge(pfr_rush_rolling, on=['season', 'week', 'team'])


#Lets rank these columns
columns_to_rank = [i for i in pfr_agg_rush.columns if i not in ['season', 'week', 'team']]

for col in columns_to_rank:
    rank_col_name = f'rank_{col}_weekly'
    pfr_agg_rush[rank_col_name] = pfr_agg_rush.groupby(['season', 'week'])[col].rank(ascending=False)

#Lets get sums/avg up to each part of the season
colums_to_expand_sum = ['carries',
 'rushing_yards_before_contact',
 'rushing_yards_after_contact',
 'rushing_broken_tackles',
 'carries_allowed',
 'rushing_yards_before_contact_allowed',
 'rushing_yards_after_contact_allowed',
 'rushing_broken_tackles_allowed']

for col in colums_to_expand_sum:
    expand_col_name = f'expand_{col}'
    pfr_agg_rush[expand_col_name] = pfr_agg_rush.sort_values(by='week').groupby(['season', 'team'])[col].cumsum()

#Lets check this is working
    
#Great, lets do this for averages
colums_to_expand_avg = ['rushing_yards_before_contact_avg',
 'rushing_yards_before_contact_avg_allowed',
 'rushing_yards_after_contact_avg_allowed',
 'rushing_yards_after_contact_avg']

for col in colums_to_expand_avg:
    expand_col_name = f'expand_{col}'
    cumulative_sum = pfr_agg_rush.sort_values(by='week').groupby(['season', 'team'])[col].cumsum()
    cumulative_count = pfr_agg_rush.sort_values(by='week').groupby(['season', 'team'])[col].cumcount() + 1
    pfr_agg_rush[expand_col_name] = cumulative_sum / cumulative_count

#Now we'll get the rank of the expanding columns

for col in pfr_agg_rush.columns:
    if 'expand_' in col:
        rank_col_name = f'rank_{col}_weekly'
        pfr_agg_rush[rank_col_name] = pfr_agg_rush.groupby(['season', 'week'])[col].rank(ascending=False)

#Now let's join to weekly agg
weekly_agg = weekly_agg.merge(pfr_agg_rush, on=['season', 'week', 'team'])


# # Injury Data

inj = nfl.import_injuries(years)
inj = inj[inj['game_type'] == 'REG'].reset_index(drop=True)

#The table lists all players on a team, even if the players aren't injured

#Filter for those who are out
#These are injuries keeping players off the field


inj = inj[inj['report_status'] == 'Out'].reset_index(drop=True)

#Changing raiders to LV
inj.loc[inj['team'] == 'OAK', 'team'] = 'LV'

#Seaon should be an integer
inj['season'] = inj['season'].astype('int')

inj_agg = inj.groupby(['season', 'week', 'team'])['gsis_id'].count().reset_index()
inj_agg = inj_agg.rename(columns={'gsis_id':'players_injured'})

#These two won't equal because some teams won't have any injuries
#When joining, need to reset the column to be 0 if null

weekly_agg = weekly_agg.merge(inj_agg, on=['season', 'week', 'team'], how='left')
weekly_agg['players_injured_adj'] = weekly_agg['players_injured'].apply(lambda x: 0 if pd.isna(x) else x)

weekly_agg = weekly_agg.drop('players_injured', axis=1)


# # QBR Data

#Get QBR data
#Data source is outdated so the creator provided a workaround

url = "https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_week_level.parquet"
qbr = pd.read_parquet(url).loc[lambda x: x.season == 2024].reset_index(drop=True)

#Changing raiders to LV
#Also in this table, LA rams are LAR, but in other tables they're just LA
#Same for washington


qbr.loc[qbr['team_abb'] == 'OAK', 'team_abb'] = 'LV'
qbr.loc[qbr['opp_abb'] == 'OAK', 'opp_abb'] = 'LV'

qbr.loc[qbr['team_abb'] == 'LAR', 'team_abb'] = 'LA'
qbr.loc[qbr['opp_abb'] == 'LAR', 'opp_abb'] = 'LA'

qbr.loc[qbr['team_abb'] == 'WSH', 'team_abb'] = 'WAS'
qbr.loc[qbr['opp_abb'] == 'WSH', 'opp_abb'] = 'WAS'


#Some teams are repeated b/c they'll play multiple QBs in a game
#So we'll take an weighted avg of each team's metrics (QBR, epa_total, etc)

qbr['share_of_plays'] = qbr['qb_plays'] / qbr.groupby(['season', 'team', 'game_week'])['qb_plays'].transform('sum')


for i in qbr.iloc[:,9:19].columns:
    col_name = f'{i}_adj'
    qbr[col_name] = qbr[i] * qbr['share_of_plays']

#This adjustment makes sense because not all QB play is equal
#A backup coming in during the 4th quarter when in the lead is different from a starter in the 1st quarter
#This being said, we'll grab an adj version and a maximum of each column
#This way we'll have the best QB performance and a weighted "team" qb performance


qbr_agg = qbr.groupby(['season', 'team_abb', 'game_week']).agg({'qbr_total':'max',
                                                            'qbr_total_adj':'sum',
                                                             'pts_added':'max',
                                                             'pts_added_adj':'sum',
                                                             'qb_plays':'max',
                                                             'qb_plays_adj':'sum',
                                                             'epa_total':'max',
                                                             'epa_total_adj':'sum',
                                                             'pass':'max',
                                                             'pass_adj':'sum',
                                                             'run':'max',
                                                             'run_adj':'sum',
                                                             'exp_sack':'max',
                                                             'exp_sack_adj':'sum',
                                                             'penalty':'max',
                                                             'penalty_adj':'sum',
                                                             'qbr_raw':'max',
                                                             'qbr_raw_adj':'sum',
                                                             'sack':'max',
                                                             'sack_adj':'sum'}).reset_index()


#Let's get QBR allowed columns now
qbr_agg_def = qbr.groupby(['season', 'opp_abb', 'game_week']).agg({'qbr_total':'max',
                                                            'qbr_total_adj':'sum',
                                                             'pts_added':'max',
                                                             'pts_added_adj':'sum',
                                                             'qb_plays':'max',
                                                             'qb_plays_adj':'sum',
                                                             'epa_total':'max',
                                                             'epa_total_adj':'sum',
                                                             'pass':'max',
                                                             'pass_adj':'sum',
                                                             'run':'max',
                                                             'run_adj':'sum',
                                                             'exp_sack':'max',
                                                             'exp_sack_adj':'sum',
                                                             'penalty':'max',
                                                             'penalty_adj':'sum',
                                                             'qbr_raw':'max',
                                                             'qbr_raw_adj':'sum',
                                                             'sack':'max',
                                                             'sack_adj':'sum'}).reset_index()


qbr_agg_def.columns = [i+'_allowed' if i not in ['season', 'team_abb', 'game_week', 'opp_abb']\
                      else i for i in qbr_agg.columns]

#Lets join the def and offensive qbr tables

qbr_agg = qbr_agg.sort_values(by=['season','team_abb','game_week']).reset_index(drop=True)
qbr_agg_def = qbr_agg_def.sort_values(by=['season','team_abb','game_week']).reset_index(drop=True)

qbr_agg = qbr_agg_def.merge(qbr_agg, on=['season','team_abb','game_week'], how='outer')

#Sometimes QB play is so poor, that QBR stats aren't recorded/calculated
#In these instances, the team didn't have QBR, and the opposing team didn't have QBR allowed
#This means some teams aren't in the qbr table, requiring an outer join
#In these instances, we'll add a 0 for those rows

qbr_agg = qbr_agg.sort_values(by=['season','team_abb','game_week']).reset_index(drop=True)

#Viewing the rows with null QBR values
#Again this is due to poor QB performance

#QBR scales from 0 to 100
#To me, if your play was so bad a QBR couldn't be calculated, I'm giving you a 0
#This also means you added no expected points, so your epa should also be a 0
#Same logic for the other columns

qbr_agg = qbr_agg.fillna(0)

#Lets get a rolling avg of these metrics

original_week = qbr_agg['game_week'].copy()

columns_to_roll = [i for i in qbr_agg.columns if i not in ['season', 'team_abb', 'game_week']]

qbr_rolling = qbr_agg.groupby(['season', 'team_abb'])[columns_to_roll].rolling(3).mean().reset_index()

qbr_rolling = qbr_rolling.drop('level_2', axis=1)

qbr_rolling['game_week'] = original_week

#Lets add "rolling" to the column names
qbr_rolling.columns = [i+'_rolling' if i not in ['season', 'team_abb', 'game_week'] \
                      else i for i in qbr_rolling.columns]


#Lets join to qbr_agg
qbr_agg = qbr_agg.merge(qbr_rolling, on=['season', 'game_week', 'team_abb'], how='outer')

#Lets get rankings of these columns
columns_to_rank = [i for i in qbr_agg.columns if i not in ['season', 'game_week', 'team_abb']]

for col in columns_to_rank:
    rank_col_name = f'rank_{col}_weekly'
    qbr_agg[rank_col_name] = qbr_agg.groupby(['season', 'game_week'])[col].rank(ascending=False)


#Lets get season-long ranks
#These are indices, so the columns will be expanded based on their average over the season

colums_to_expand_avg = ['qbr_total_allowed',
 'qbr_total_adj_allowed',
 'pts_added_allowed',
 'pts_added_adj_allowed',
 'qb_plays_allowed',
 'qb_plays_adj_allowed',
 'epa_total_allowed',
 'epa_total_adj_allowed',
 'pass_allowed',
 'pass_adj_allowed',
 'run_allowed',
 'run_adj_allowed',
 'exp_sack_allowed',
 'exp_sack_adj_allowed',
 'penalty_allowed',
 'penalty_adj_allowed',
 'qbr_raw_allowed',
 'qbr_raw_adj_allowed',
 'sack_allowed',
 'sack_adj_allowed',
 'qbr_total',
 'qbr_total_adj',
 'pts_added',
 'pts_added_adj',
 'qb_plays',
 'qb_plays_adj',
 'epa_total',
 'epa_total_adj',
 'pass',
 'pass_adj',
 'run',
 'run_adj',
 'exp_sack',
 'exp_sack_adj',
 'penalty',
 'penalty_adj',
 'qbr_raw',
 'qbr_raw_adj',
 'sack',
 'sack_adj',]


for col in colums_to_expand_avg:
    expand_col_name = f'expand_{col}'
    cumulative_sum = qbr_agg.sort_values(by='game_week').groupby(['season', 'team_abb'])[col].cumsum()
    cumulative_count = qbr_agg.sort_values(by='game_week').groupby(['season', 'team_abb'])[col].cumcount() + 1
    qbr_agg[expand_col_name] = cumulative_sum / cumulative_count


#Now we'll get the rank of the expanding columns
for col in qbr_agg.columns:
    if 'expand_' in col:
        rank_col_name = f'rank_{col}_weekly'
        qbr_agg[rank_col_name] = qbr_agg.groupby(['season', 'game_week'])[col].rank(ascending=False)


#Lets join to weekly_agg
qbr_agg = qbr_agg.rename(columns={'game_week':'week',
                                'team_abb':'team'})

weekly_agg = weekly_agg.merge(qbr_agg, on=['season', 'week', 'team'], how='left')


# # Game Info

info = nfl.import_schedules(years)
info = info[info['game_type'] == 'REG'].reset_index(drop=True)

#Let's reduce this to only the games that have been played or are coming this week
info = info[info['week'] <= weekly_agg['week'].max() + 1].reset_index(drop=True)

#Lets see some of the odds info

#Changing raiders info

info.loc[info['away_team'] == 'OAK', 'away_team'] = 'LV'
info.loc[info['home_team'] == 'OAK', 'home_team'] = 'LV'


odds = info[['season', 'week', 'away_team', 'home_team', 'away_moneyline', 
      'home_moneyline', 'spread_line', 'away_spread_odds',
       'home_spread_odds', 'total_line', 'under_odds', 'over_odds',
            'home_score', 'away_score']]

#Comapring the spread_line to the historical odds site, spread_line is the away team spread
#So we can get home team spread by multiplying by -1 to get the inverse

odds = odds.rename(columns={'spread_line':'away_spread'})

odds['home_spread'] = odds['away_spread'] * -1

odds['home_diff'] = odds['home_score'] - odds['away_score']
odds['away_diff'] = odds['away_score'] - odds['home_score']

odds['home_cover'] = odds['home_spread'] + odds['home_diff']
odds['away_cover'] = odds['away_spread'] + odds['away_diff']

#Lets also get the total points
odds['total_score'] = odds['home_score'] + odds['away_score']

#We can group by season, week, away team (away stats)
#Then we can group by season, week, home team (home stats)

#Then we can concat and change the column names to have a betting table

bet_away = odds[['season', 'week', 'away_team','away_moneyline',
                               'away_spread',
                               'away_spread_odds',
                                 'total_line',
                                 'under_odds',
                                 'over_odds',
                                 'away_cover',
                                 'home_cover',
                                  'away_score',
                                 'away_diff',
                                'home_score',
                                 'home_diff',
                                 'total_score']]


bet_away = bet_away.rename(columns={'away_team':'team',
                                   'away_moneyline':'moneyline',
                                    'away_spread':'spread',
                                   'away_spread_odds':'spread_odds',
                                   'total_line':'exp_total_points',
                                   'away_cover':'cover',
                                    'home_cover':'cover_allowed',
                                   'away_score':'points_scored',
                                    'away_diff':'scoring_margin',
                                'home_score':'points_allowed',
                                 'home_diff':'scoring_margin_allowed',
                                   'total_score':'total_points_scored'})


bet_away['is_home'] = 0

bet_home = odds[['season', 'week', 'home_team','home_moneyline',
                               'home_spread',
                               'home_spread_odds',
                                 'total_line',
                                 'under_odds',
                                 'over_odds',
                                 'home_cover',
                                 'away_cover',
                                  'home_score',
                                 'home_diff',
                                  'away_score',
                                 'away_diff',
                                 'total_score']]



bet_home = bet_home.rename(columns={'home_team':'team',
                                   'home_moneyline':'moneyline',
                                    'home_spread':'spread',
                                   'home_spread_odds':'spread_odds',
                                   'total_line':'exp_total_points',
                                   'home_cover':'cover',
                                    'away_cover':'cover_allowed',
                                   'home_score':'points_scored',
                                    'home_diff':'scoring_margin',
                                  'away_score':'points_allowed',
                                 'away_diff':'scoring_margin_allowed',
                                   'total_score':'total_points_scored'})


bet_home['is_home'] = 1

bet = pd.concat([bet_home, bet_away])

bet = bet.sort_values(by=['season', 'team', 'week']).reset_index(drop=True)

#Lets get rolling avg points scored, points allowed, scoring margin, margin allowed, spread, cover, cover allowed 
#Then we'll get these metrics up to that point in the season
#Then of course, we rank


original_week = bet['week'].copy()
#Since there's information for the upcoming week, let's not get any rolling information here

bet_rolling = bet.groupby(['season', 'team'])[['points_scored', 
                                              'points_allowed',
                                              'scoring_margin',
                                              'scoring_margin_allowed', 
                                              'spread',
                                              'cover', 
                                              'cover_allowed']].rolling(3).mean().reset_index()

bet_rolling = bet_rolling.drop('level_2', axis=1)

bet_rolling['week'] = original_week

#Lets change the column names
bet_rolling.columns = [i+'_rolling' if i not in ['season', 'week', 'team'] else i for i in bet_rolling.columns]


#Let's join to bet
bet = bet.merge(bet_rolling, on=['season', 'team', 'week'])

#Lets get the metrics we rolled, but expanding across the season
for col in ['points_scored', 
              'points_allowed',
              'scoring_margin',
              'scoring_margin_allowed', 
              'spread',
              'cover', 
              'cover_allowed']:
    expand_col_name = f'expand_{col}'
    bet[expand_col_name] = bet.sort_values(by='week').groupby(['season', 'team'])[col].cumsum()


#Now lets rank these columns
for col in [i for i in bet.columns if 'expand_' in i or '_rolling' in i]:
    rank_col_name = f'rank_{col}_weekly'
    bet[rank_col_name] = bet.groupby(['season', 'week'])[col].rank(ascending=False)


#Now we have a table w betting info
#This table also has items regarding stadium surface, division game, etc
#Lets get this info but split between the away and home team

info_away = info[['season', 'week', 'away_team', 'away_rest', 'overtime',
     'div_game', 'roof', 'surface']]

info_away = info_away.rename(columns={'away_team':'team',
                                     'away_rest':'rest'})
info_home = info[['season', 'week', 'home_team', 'home_rest', 'overtime',
     'div_game', 'roof', 'surface']]


info_home = info_home.rename(columns={'home_team':'team',
                                     'home_rest':'rest'})


#Since roof and surface are categorical, lets create dummy variables
info_away_surface = pd.get_dummies(info_away['surface'])

#The historical dataset has both 'grass' AND 'grass ' as an options
#I'll have to see if this is an error in historical data, or something in the ETL
#Creating a 'grass ' column for now

info_away_surface['grass '] = 0

info_away_roof = pd.get_dummies(info_away['roof'])


#For the 2024 season, there aren't any stadiums categorized as "open"
#This could be due to a change in definition, but for now let's add "open" as a columns

info_away_roof['open'] = 0


info_away = pd.concat([info_away, info_away_surface, info_away_roof], axis=1)


info_home_surface = pd.get_dummies(info_home['surface'])


#Same reasoninig as the away surface
info_home_surface['grass '] = 0


info_home_roof = pd.get_dummies(info_home['roof'])

#Same reasoninig as the away roofs
info_home_roof['open'] = 0


info_home = pd.concat([info_home, info_home_surface, info_home_roof], axis=1)


#Lets combine the away and home info
weekly_info = pd.concat([info_home, info_away])
weekly_info = weekly_info.sort_values(by=['season', 'team', 'week']).reset_index(drop=True)

#We dont need roof and surface columns since we got the dummy columns
weekly_info = weekly_info.drop(['roof', 'surface'], axis=1)

#Let's combine bet and weekly_info to weekly_agg

#Weekly_agg only has info for games that occured
#As a result, we need to left join to the bet table so that we have info on the upcoming game
#We'll have to shift the columns from the prior week so for the upcoming week we see prior game(s) data

weekly_agg = bet.merge(weekly_agg, on=['season', 'week', 'team'], how='left')

#Lets join weekly info
weekly_agg = weekly_agg.merge(weekly_info, on=['season', 'week', 'team'])

#Now let's create some binary variables that we could use as target variables in future models

#First, lets make a column for whether a team covered the spread
weekly_agg['is_cover'] = weekly_agg['cover'].apply(lambda x: 1 if x > 0 else 0)

#Now let's get a column for whether a team won the game
weekly_agg['is_winner'] = weekly_agg['scoring_margin'].apply(lambda x: 1 if x > 0 else 0)

#Now let's get a column for whether over under was hit
weekly_agg['is_over'] = (weekly_agg['total_points_scored'] > weekly_agg['exp_total_points']).astype('int')

#Now lets get whether they're favored
weekly_agg['is_fav'] = np.where(weekly_agg['spread'] < 0, 1, 0)

#Now let's get their "record" both ATS and Straight Up, as well as record for over/under and being favored
#We can do this by getting an avg of is_cover, is_winner,and is_over, and is_fav

for col in ['is_cover', 'is_winner', 'is_over', 'is_fav']:
    expand_col_name = f'{col}_record'
    cumulative_sum = weekly_agg.sort_values(by='week').groupby(['season', 'team'])[col].cumsum()
    cumulative_count = weekly_agg.sort_values(by='week').groupby(['season', 'team'])[col].cumcount() + 1
    weekly_agg[expand_col_name] = cumulative_sum / cumulative_count

#Lets check that this worked

#Why not get a rank of these columns as well?
for col in ['is_cover_record', 'is_winner_record', 'is_over_record', 'is_fav_record']:
    rank_col_name = f'rank_{col}'
    weekly_agg[rank_col_name] = weekly_agg.groupby(['season', 'week'])[col].rank(ascending=False)


#I realize that rushing carries were in weekly_data as well as in pfr_rush
#Lets identify them

weekly_agg = weekly_agg.drop([i for i in weekly_agg.columns if i.endswith(('_y'))], axis=1)

#Lets remove _x from the remaining columns
weekly_agg.columns = [i for i in weekly_agg.columns.str.replace('_x', '')]


#Let's shift the data so that for the current week, we see previous game data
columns_to_keep = ['rest', 'overtime', 'div_game', 'a_turf', 'astroturf', 'fieldturf',
       'grass', 'matrixturf', 'sportturf', 'closed', 'dome', 'outdoors', 'open',
       'is_cover', 'is_winner', 'is_over', 'is_fav', 'is_home','season', 'week', 'team', 
        'moneyline', 'spread', 'spread_odds','exp_total_points', 'under_odds', 'over_odds',
        'players_injured_adj']


for i in weekly_agg.columns:
    if i not in columns_to_keep:
        weekly_agg[i+'_shifted'] = weekly_agg.groupby(['season', 'team'])[i].shift(periods=1)
    else:
        continue



#This is working as expected
#We're keeping the shifted columns since that's bringin the up to date data for the upcoming week

weekly_agg = weekly_agg[[i for i in weekly_agg.columns if '_shifted' in i or i in columns_to_keep]]


#We need to move injuries portion to the end since this is upcoming game information
#Issue is that if we shift this column, this will affect the order of the data
#Which will throw off the prediction too much

weekly_agg['players_injured_adj'] = weekly_agg['players_injured_adj'].fillna(0)


#Now we can use pandas to write this file to S3

#Let's pull our S3 Credentials, and specify the S3 bucket we wish to write to

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

aws_s3_bucket = "nfl.data"

#Now we write to S3

weekly_agg.to_csv(
    f"s3://{aws_s3_bucket}/nfl_current_data.csv",
    index=False,
    storage_options={
        "key": aws_access_key_id,
        "secret": aws_secret_access_key,
        }
)

#We can also write this to our github folder

weekly_agg.to_csv('PredictingNFLGames/nfl_current_data.csv')




