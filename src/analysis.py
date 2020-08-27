__author__ = "Abhay Rajendra Dixit "
__author__ = "Pranjal Pandey"
__author__ = "Ravikiran Jois Yedur Prabhakar"

import datetime
import json
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None  # default='warn'
from pymongo import MongoClient
import matplotlib.lines
import pingouin as pg
import numpy as np
import json


def get_mongo_params(file):
    """
         This function reads the files and initialise mongoDB configuration
         for the project
         :param file: file path of the config file
         :return filePath dictionary: The dictionary of connection information like port, host and database.
    """
    with open(file, 'r') as f:
        source = json.loads(f.read())
        return dict(
            host=source['host'],
            port=source['port'],
            database=source['database']
        )


def get_files_path_params(file):
    """
         This function reads file path of three data-sets from the config files
         :param file: file path of the config file
         :return filePath dictionary: The dictionary of all the source file paths
    """
    with open(file, 'r') as f:
        source = json.loads(f.read())

        return dict(
            traffic_crash_data_path=str(source['traffic_crash_data_path']),
            redlight_data_path=str(source['redlight_data_path']),
            speed_violations_data_path=str(source['speed_violations_data_path']),
        )


def get_mongo_connection(mongo_dict):
    """
         This function reads connection dictionary file and establish the
         connection to mongoDb database
         :param mongo_dict: The dictionary of all the source file paths
         :return collection: Mongodb connection string which contains host and port information
    """
    try:
        connection = MongoClient("mongodb://" + mongo_dict['host'] + ":" + str(mongo_dict['port']))
        print("Connected to MongoDb successfully!")
    except:
        print("Could not connect to MongoDB")
    return mongo_dict["database"], connection


def time_series_analysis_combined(traffic_analysis, mongo_con):
    """
        Method to display combined information of the time series data for each collection i.e., Speed Camera Violations,
        Traffic Camera violations and Red Light Violation
        :param traffic_analysis: the name of the database on mongoDB
        :param mongo_conn: mongoDB connection object
        :return: None
        """
    db = mongo_con[traffic_analysis]

    """With Speed"""
    speed_time = list(db.speed.aggregate([
        {'$project': {
            "VIOLATIONS": 1,
            "month": {"$month": "$VIOLATION DATE"}
        }},
        {'$group': {
                '_id': "$month",
                'total': {"$sum": "$VIOLATIONS"}
            }
        },
        {'$sort': {'_id': 1}}
    ]))
    print(speed_time)
    months = []
    violations = []
    for item in speed_time:
        months.append(item['_id'])
        violations.append(item['total'])

    plt.fill_between(months, violations, color="orange", label='Speed Camera')
    plt.xticks(months)
    # plt.show()

    """With Violations"""
    violation_coll = list(db.violation.aggregate([
        {'$project': {
            "VIOLATIONS": 1,
            "month": {"$month": "$VIOLATION DATE"}
        }},
        {'$group': {
            '_id': "$month",
            'total': {"$sum": "$VIOLATIONS"}
            }
        },
        {'$sort': {'_id': 1}}
    ]))
    print(violation_coll)
    months = []
    violations = []
    for item in violation_coll:
        months.append(item['_id'])
        violations.append(item['total'])

    plt.fill_between(months, violations, color='red', label='Red Light')
    plt.xticks(months)
    # plt.show()

    """With Traffic Violations"""
    traffic_crash = list(db.traffic_crash.aggregate([
        {'$project':
            {"month": {"$month": "$Date"}}
        },
        {'$group': {
            '_id': "$month",
            'total': {"$sum": 1}
            }
        },
        {'$sort': {'_id': 1}}
    ]))
    print(traffic_crash)
    months = []
    violations = []
    for item in traffic_crash:
        months.append(item['_id'])
        violations.append(item['total'])

    plt.fill_between(months, violations, color="skyblue", label='Traffic Crashes')
    plt.xticks(months)
    plt.legend(loc="upper right")
    plt.xlabel("Month")
    plt.ylabel("No. of Violations")
    plt.show()


def time_series_analysis_separated(traffic_analysis, mongo_conn):
    """
    Method to display the time series data for each collection i.e., Speed Camera Violations, Traffic Camera violations
    and Red Light Violation separately
    :param traffic_analysis: the name of the database on mongoDB
    :param mongo_conn: mongoDB connection object
    :return: None
    """
    month_long_to_short = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep",
                           10: "Oct", 11: "Nov", 12: "Dec"}
    year_long_to_short = {2015: '15', 2016: '16', 2017: '17', 2018: '18', 2019: '19', 2020: '20'}
    db = mongo_conn[traffic_analysis]

    """Speed Collection"""
    speed_data = list(db.speed.aggregate([
                        {'$group': {
                                '_id': {
                                    'year': {'$year': '$VIOLATION DATE'},
                                    'month': {'$month': '$VIOLATION DATE'}
                                },
                                'total': {'$sum': '$VIOLATIONS'}}
                        }, {'$sort': {'_id': 1}}
                    ]))
    months = []
    years = []
    violations = []
    for item in speed_data:
        months.append(item['_id']['month'])
        years.append(item['_id']['year'])
        violations.append(item['total'])

    speed = {}
    speed['Month'] = months
    speed['Year'] = years
    speed['Violations'] = violations

    speed_df = pd.DataFrame(speed)
    speed_df['Month'] = speed_df['Month'].map(month_long_to_short)
    speed_df['Year'] = speed_df['Year'].map(year_long_to_short)
    # speed_df['Year'] = speed_df['Year'].astype(str)
    speed_df['Month'] = speed_df['Month'].astype(str)
    speed_df['Month_Year'] = speed_df['Month'] + " '" + speed_df['Year']
    speed_df['Moving Averages'] = speed_df.rolling(window=6).mean()
    ax_speed = speed_df.set_index('Month_Year')['Violations'].plot(kind='line', figsize=(20, 10), color='purple', rot=90,
                                                                   label="Speed Camera Violations", grid=True)
    speed_df.set_index('Month_Year')['Moving Averages'].plot(kind='line', figsize=(20, 10), color='cadetblue',
                                                                   rot=90, label="Simple Moving Average (Violations)", grid=True)
    # speed_df.set_index('Month_Year')['Violations'].plot(kind='bar', figsize=(20, 10), color='cadetblue',
    #                                                                rot=90, position=0.1, label="Speed Camera Violations")
    ax_speed.set_xticks(speed_df.index)
    ax_speed.set_xticklabels(speed_df['Month_Year'], rotation=90)

    plt.legend(loc="upper right")
    plt.title("Speed Camera Violation vs. Month")
    plt.xlabel("Month")
    plt.ylabel("No. of Violations")
    plt.show()

    """Red Light Violations Collection"""
    red_light = list(db.violation.aggregate([
        {'$group': {
            '_id': {
                'year': {'$year': '$VIOLATION DATE'},
                'month': {'$month': '$VIOLATION DATE'}
            },
            'total': {'$sum': '$VIOLATIONS'}}
        }, {'$sort': {'_id': 1}}
    ]))
    months = []
    years = []
    violations = []
    for item in red_light:
        months.append(item['_id']['month'])
        years.append(item['_id']['year'])
        violations.append(item['total'])

    red_light_dict = {}
    red_light_dict['Month'] = months
    red_light_dict['Year'] = years
    red_light_dict['Violations'] = violations

    red_light_df = pd.DataFrame(red_light_dict)
    red_light_df['Month'] = red_light_df['Month'].map(month_long_to_short)
    red_light_df['Year'] = red_light_df['Year'].map(year_long_to_short)
    # red_light_df['Year'] = red_light_df['Year'].astype(str)
    red_light_df['Month'] = red_light_df['Month'].astype(str)
    red_light_df['Month_Year'] = red_light_df['Month'] + " '" + red_light_df['Year']
    red_light_df['Moving Averages'] = red_light_df.rolling(window=6).mean()
    ax_red_light = red_light_df.set_index('Month_Year')['Violations'].plot(kind='line', figsize=(20, 10), color='red',
                                                                           rot=90, label="Red Light Violations", grid=True)
    red_light_df.set_index('Month_Year')['Moving Averages'].plot(kind='line', figsize=(20, 10), color='black', rot=90,
                                                            label="Simple Moving Average (Violations)", grid=True)

    ax_red_light.set_xticks(speed_df.index)
    ax_red_light.set_xticklabels(speed_df['Month_Year'], rotation=90)

    plt.legend(loc="upper right")
    plt.title("Red Light Violation vs. Month")
    plt.xlabel("Month")
    plt.ylabel("No. of Violations")
    plt.show()

    """Traffic Crashes Collection"""
    traffic_crash = list(db.traffic_crash.aggregate([
        {'$group': {'_id': {
                        'year': {'$year': '$Date'},
                        'month': {'$month': '$Date'}
                      },
                      'total': {'$sum': 1}}
        }, {'$sort': {'_id': 1}}
    ]))
    months = []
    years = []
    violations = []
    for item in traffic_crash:
        months.append(item['_id']['month'])
        years.append(item['_id']['year'])
        violations.append(item['total'])

    traffic_dict = {}
    traffic_dict['Month'] = months
    traffic_dict['Year'] = years
    traffic_dict['Violations'] = violations

    traffic_crash_df = pd.DataFrame(traffic_dict)
    traffic_crash_df['Month'] = traffic_crash_df['Month'].map(month_long_to_short)
    traffic_crash_df['Year'] = traffic_crash_df['Year'].map(year_long_to_short)

    traffic_crash_df['Month'] = traffic_crash_df['Month'].astype(str)
    traffic_crash_df['Month_Year'] = traffic_crash_df['Month'] + " '" + traffic_crash_df['Year']
    traffic_crash_df['Moving Averages'] = traffic_crash_df.rolling(window=6).mean()
    ax_traffic_crash = traffic_crash_df.set_index('Month_Year')['Violations'].plot(kind='line', figsize=(20, 10), color='green',
                                                                           rot=90, label="Traffic Crashes", grid=True)
    traffic_crash_df.set_index('Month_Year')['Moving Averages'].plot(kind='line', figsize=(20, 10), color='orange', rot=90,
                                                            label="Simple Moving Average (Violations)", grid=True)

    ax_traffic_crash.set_xticks(speed_df.index)
    ax_traffic_crash.set_xticklabels(speed_df['Month_Year'], rotation=90)

    plt.legend(loc="upper right")
    plt.title("Traffic Crashes vs. Month")
    plt.xlabel("Month")
    plt.ylabel("No. of Violations")
    plt.show()


def time_series_analysis_red_deseasoning(traffic_analysis, mongo_conn):
    """
    Method to display the time series data for Red Light Violation separately
    :param traffic_analysis: the name of the database on mongoDB
    :param mongo_conn: mongoDB connection object
    :return: None
    """
    month_long_to_short = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep",
                           10: "Oct", 11: "Nov", 12: "Dec"}
    year_long_to_short = {2015: '15', 2016: '16', 2017: '17', 2018: '18', 2019: '19', 2020: '20'}
    db = mongo_conn[traffic_analysis]

    """Red Light Violations Collection"""
    red_light = list(db.violation.aggregate([
        {'$group': {
            '_id': {
                'year': {'$year': '$VIOLATION DATE'},
                'month': {'$month': '$VIOLATION DATE'}
            },
            'total': {'$sum': '$VIOLATIONS'}}
        }, {'$sort': {'_id': 1}}
    ]))

    months = []
    years = []
    violations = []
    for item in red_light:
        months.append(item['_id']['month'])
        years.append(item['_id']['year'])
        violations.append(item['total'])

    red_light_dict = {}
    red_light_dict['Month'] = months
    red_light_dict['Year'] = years
    red_light_dict['Violations'] = violations

    red_light_df = pd.DataFrame(red_light_dict)

    red_light_seasoner = red_light_df['Violations'].groupby([red_light_df.Year, red_light_df.Month], sort=False).sum().unstack()
    red_light_seasoner_original = copy.copy(red_light_seasoner)
    red_light_seasoner_mean_rows = red_light_seasoner.mean(axis=1)
    for index in range(2015, 2021):
        for col in red_light_seasoner:
            red_light_seasoner[col][index] = red_light_seasoner[col][index] / red_light_seasoner_mean_rows[index]

    red_light_seasoner_mean_cols = red_light_seasoner.mean(axis=0)
    index = 1
    for col in red_light_seasoner:
        # for index in range(len(red_light_seasoner)):
        red_light_seasoner[col] = red_light_seasoner_original[col] / red_light_seasoner_mean_cols[index]
        index += 1

    red_light_seasoner = red_light_seasoner.unstack().reset_index()
    red_light_seasoner.rename(columns={0: 'Violations'}, inplace=True)
    columns_titles = ["Year", "Month", "Violations"]
    red_light_seasoner = red_light_seasoner.reindex(columns=columns_titles)
    df_2015 = red_light_seasoner[red_light_seasoner['Year'] == 2015]
    df_2016 = red_light_seasoner[red_light_seasoner['Year'] == 2016]
    df_2017 = red_light_seasoner[red_light_seasoner['Year'] == 2017]
    df_2018 = red_light_seasoner[red_light_seasoner['Year'] == 2018]
    df_2019 = red_light_seasoner[red_light_seasoner['Year'] == 2019]
    df_2020 = red_light_seasoner[red_light_seasoner['Year'] == 2020]
    red_light_seasoner = pd.concat([df_2015, df_2016, df_2017, df_2018, df_2019, df_2020])

    red_light_df['Month'] = red_light_df['Month'].map(month_long_to_short)
    red_light_df['Year'] = red_light_df['Year'].map(year_long_to_short)
    # red_light_df['Year'] = red_light_df['Year'].astype(str)
    red_light_df['Month'] = red_light_df['Month'].astype(str)
    red_light_df['Month_Year'] = red_light_df['Month'] + " '" + red_light_df['Year']
    red_light_seasoner['Month'] = red_light_seasoner['Month'].map(month_long_to_short)
    red_light_seasoner['Year'] = red_light_seasoner['Year'].map(year_long_to_short)
    # red_light_df['Year'] = red_light_df['Year'].astype(str)
    red_light_seasoner['Month'] = red_light_seasoner['Month'].astype(str)
    red_light_seasoner['Month_Year'] = red_light_seasoner['Month'] + " '" + red_light_seasoner['Year']

    ax_red_light = red_light_df.set_index('Month_Year')['Violations'].plot(kind='line', figsize=(20, 10), color='red',
                                                                rot=90, label="Red Light Violations", grid=True)
    red_light_seasoner.set_index('Month_Year')['Violations'].plot(kind='line', figsize=(20, 10), color='black',
                                                                rot=90, label="Deseasoned Violations (Violations)", grid=True)

    ax_red_light.set_xticks(red_light_df.index)
    ax_red_light.set_xticklabels(red_light_df['Month_Year'], rotation=90)

    plt.legend(loc="upper right")
    plt.title("Red Light Violation vs. Month")
    plt.xlabel("Month")
    plt.ylabel("No. of Violations")
    plt.show()


def visualize(x_df, y_df, x_label, y_label, cor, color):
    """
    Method to display graphs depicting correlation between red light violations, speed violations and total violation
    against crashes with respect to date and location.
    :param x_df: dataframe as x-axis
    :param y_df: dataframe as y-axis
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param cor: correlation coefficient to be displayed
    :param color: color of the curve
    :return: None
    """
    f1 = plt.figure()
    plt.scatter(x_df, y_df, label="Pearson's Correlation Coefficient = {:.3f}".format(cor.r.pearson), color=color, s=15, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " v/s " + y_label)
    plt.legend()
    plt.draw()
    return f1


def date_perspective(speed_df, red_light_df, traffic_crash_df):
    """
    Method to calculate and display correlation between redlight violations, speed violations and total violation against crashes with
    respect to date.
    :param speed_df: dataframe consisting of speed violation data
    :param red_light_df: dataframe consisting of red light violation data
    :param traffic_crash_df: dataframe consisting of traffic crash data
    :return: None
    """


    date_red_light_frame = red_light_df[['VIOLATION DATE', 'VIOLATIONS']]
    date_speed_frame = speed_df[['VIOLATION DATE', 'VIOLATIONS']]
    date_traffic_crash = traffic_crash_df[["Date"]]
    date_red_light_frame = date_red_light_frame.groupby('VIOLATION DATE', sort=False, as_index=False)[
        'VIOLATIONS'].sum()
    date_speed_frame = date_speed_frame.groupby('VIOLATION DATE', sort=False, as_index=False)['VIOLATIONS'].sum()
    date_traffic_crash['count'] = date_traffic_crash.groupby('Date')['Date'].transform('count')
    date_traffic_crash.rename(columns={'count': 'Crashes'}, inplace=True)
    date_speed_frame.rename(columns={'VIOLATION DATE': 'Date', 'VIOLATIONS': 'Red_Light_Violations'}, inplace=True)
    date_red_light_frame.rename(columns={'VIOLATION DATE': 'Date', 'VIOLATIONS': 'Speed_Limit_Violations'},
                                inplace=True)

    res = pd.merge(date_traffic_crash, date_speed_frame, how='left', on='Date', sort=True).drop_duplicates()
    res = pd.merge(res, date_red_light_frame, how='left', on='Date', sort=True).drop_duplicates()
    res['Total_Violations'] = res['Red_Light_Violations'] + res['Speed_Limit_Violations']
    res.fillna(0, inplace=True)
    cor1 = pg.corr(x=res['Red_Light_Violations'], y=res['Crashes'])
    cor2 = pg.corr(x=res['Speed_Limit_Violations'], y=res['Crashes'])
    cor3 = pg.corr(x=res['Total_Violations'], y=res['Crashes'])

    f1 = visualize(res["Red_Light_Violations"], res["Crashes"], "Red Light Violations", "Crashes", cor1, 'blue')
    f2 = visualize(res["Speed_Limit_Violations"], res["Crashes"], "Speed-limit Violations", "Crashes", cor2, 'blue')
    f3 = visualize(res["Total_Violations"], res["Crashes"], "Total Violations", "Crashes", cor3, 'blue')

    return f1, f2, f3


def location_perspective(speed_df, red_light_df, traffic_crash_df):
    """
    Method to display correlation between red light violations, speed violations and total violation against crashes with
    respect to location.
    :param speed_df: dataframe consisting of speed violation data
    :param red_light_df: dataframe consisting of red light violation data
    :param traffic_crash_df: dataframe consisting of traffic crash data
    :return: None
    """
    speed_frame_sample = speed_df[['STREET_NAME', 'VIOLATIONS']].groupby('STREET_NAME', as_index=False).sum()
    red_light_frame_sample = red_light_df[['STREET_NAME', 'VIOLATIONS']].groupby('STREET_NAME', as_index=False).sum()
    traffic_frame_sample = traffic_crash_df[['STREET_NAME', 'Date']].groupby('STREET_NAME', as_index=False).count()

    red_light_frame_sample.columns = ["STREET_NAME", "REDLIGHT_VIOLATIONS"]
    speed_frame_sample.columns = ["STREET_NAME", "SPEED_VIOLATIONS"]
    res = pd.merge(traffic_frame_sample, speed_frame_sample, how='left', on='STREET_NAME')
    res = pd.merge(res, red_light_frame_sample, how='left', on='STREET_NAME')
    res.columns = ["STREET_NAME", "REDLIGHT_VIOLATIONS", "SPEED_VIOLATIONS", "Crashes"]
    res["Total_violations"] = res["REDLIGHT_VIOLATIONS"] + res["SPEED_VIOLATIONS"]
    res['REDLIGHT_VIOLATIONS'].fillna(0, inplace=True)
    res['SPEED_VIOLATIONS'].fillna(0, inplace=True)
    res['Crashes'].fillna(0, inplace=True)
    res['Total_violations'].fillna(0, inplace=True)

    cor1 = pg.corr(x=res['REDLIGHT_VIOLATIONS'], y=res['Crashes'])
    cor2 = pg.corr(x=res['SPEED_VIOLATIONS'], y=res['Crashes'])
    cor3 = pg.corr(x=res['Total_violations'], y=res['Crashes'])

    f1 = visualize(res["REDLIGHT_VIOLATIONS"], res["Crashes"], "Red Light Violations", "Crashes", cor1, 'red')
    f2 = visualize(res["SPEED_VIOLATIONS"], res["Crashes"], "Speed Camera Violations", "Crashes", cor2, 'orange')
    f3 = visualize(res["SPEED_VIOLATIONS"], res["Crashes"], "Total Violations", "Crashes", cor3, 'lightblue')

    return f1, f2, f3


def correlation(db_name, mongo_conn):
    """
    Method to display correlation between red light violations, speed violations and total violation against crashes with
    respect to location and date.
    :return: None
    """

    db = mongo_conn[db_name]
    speed_df = pd.DataFrame(list(db.speed.find({})))
    red_light_df = pd.DataFrame(list(db.violation.find({})))
    traffic_crash_df = pd.DataFrame(list(db.traffic_crash.find({})))

    date_perspective(speed_df, red_light_df, traffic_crash_df)

    location_perspective(speed_df, red_light_df, traffic_crash_df)


def descriptive_stats(db_name, mongo_conn):
    """
    Method to print interesting statistics of data based on date and location.
    :return: None
    """
    db = mongo_conn[db_name]
    speed_df = pd.DataFrame(list(db.speed.find({})))
    red_light_df = pd.DataFrame(list(db.violation.find({})))
    traffic_crash_df = pd.DataFrame(list(db.traffic_crash.find({})))

    speed_frame_sample = speed_df[['VIOLATION DATE', 'VIOLATIONS']].groupby('VIOLATION DATE', as_index=False).sum()
    red_light_frame_sample = red_light_df[['VIOLATION DATE', 'VIOLATIONS']].groupby('VIOLATION DATE', as_index=False).sum()
    traffic_frame_sample = traffic_crash_df[['Date', 'index']].groupby('Date', as_index=False).count()

    print("Average number of Speed Limit Violations in Chicago per day:", speed_frame_sample['VIOLATIONS'].mean())
    print("Average number of Red Light Violations in Chicago per day:", red_light_frame_sample['VIOLATIONS'].mean())
    print("Average number of Traffic Crashes in Chicago per day:", traffic_frame_sample['index'].mean())


    speed_frame_sample = speed_df[['STREET_NAME', 'VIOLATIONS']].groupby('STREET_NAME', as_index=False).sum()
    red_light_frame_sample = red_light_df[['STREET_NAME', 'VIOLATIONS']].groupby('STREET_NAME', as_index=False).sum()
    traffic_frame_sample = traffic_crash_df[['STREET_NAME', 'Date']].groupby('STREET_NAME', as_index=False).count()
    traffic_frame_sample.columns = ['STREET_NAME', 'Crashes']

    print()
    print("Streets with most number of Red Light Violations")
    print(red_light_frame_sample.sort_values(by=['VIOLATIONS'], ascending=False).head(10))

    print()
    print("Streets with most number of Speed Limit Violations")
    print(speed_frame_sample.sort_values(by=['VIOLATIONS'], ascending=False).head(10))

    print()
    print("Streets with most number of Traffic crashes")
    print(traffic_frame_sample.sort_values(by=['Crashes'], ascending=False).head(10))


def clustering_by_location(speed_df, red_light_df, traffic_crash_df):
    """
    Method to implement k-means clustering for data based on location
    :param speed_df: dataframe consisting of speed violation data
    :param red_light_df: dataframe consisting of red light violation data
    :param traffic_crash_df: dataframe consisting of traffic crash data
    :return: None
    """

    speed_frame_sample = speed_df[['STREET_NAME', 'VIOLATIONS']].groupby('STREET_NAME', as_index=False).sum()
    red_light_frame_sample = red_light_df[['STREET_NAME', 'VIOLATIONS']].groupby('STREET_NAME', as_index=False).sum()
    traffic_frame_sample = traffic_crash_df[['STREET_NAME', 'Date']].groupby('STREET_NAME', as_index=False).count()

    red_light_frame_sample.columns = ["STREET_NAME", "REDLIGHT_VIOLATIONS"]
    speed_frame_sample.columns = ["STREET_NAME", "SPEED_VIOLATIONS"]
    res = pd.merge(traffic_frame_sample, speed_frame_sample, how='left', on='STREET_NAME')
    res = pd.merge(res, red_light_frame_sample, how='left', on='STREET_NAME')
    res.columns = ["STREET_NAME", "REDLIGHT_VIOLATIONS", "SPEED_VIOLATIONS", "Crashes"]
    res["Total_violations"] = res["REDLIGHT_VIOLATIONS"] + res["SPEED_VIOLATIONS"]
    res['REDLIGHT_VIOLATIONS'].fillna(0, inplace=True)
    res['SPEED_VIOLATIONS'].fillna(0, inplace=True)
    res['Crashes'].fillna(0, inplace=True)
    res['Total_violations'].fillna(0, inplace=True)

    red_crash = res[['REDLIGHT_VIOLATIONS', 'Crashes']]
    speed_crash = res[['SPEED_VIOLATIONS', 'Crashes']]

    kmeans, y_kmeans = k_means(red_crash.to_numpy())
    k_means_visualization(red_crash.to_numpy(), kmeans, y_kmeans, "Red Light Violations", "Crashes", "Location")

    kmeans, y_kmeans = k_means(speed_crash.to_numpy())
    k_means_visualization(speed_crash.to_numpy(), kmeans, y_kmeans, "Speed Light Violations", "Crashes", "Location")


def clustering_by_date(speed_df, red_light_df, traffic_crash_df):
    """
    Method to implement k-means clustering for data based on date
    :param speed_df: dataframe consisting of speed violation data
    :param red_light_df: dataframe consisting of red light violation data
    :param traffic_crash_df: dataframe consisting of traffic crash data
    :return: None
    """

    speed_frame_sample = speed_df[['VIOLATION DATE', 'VIOLATIONS']].groupby('VIOLATION DATE', as_index=False).sum()
    red_light_frame_sample = red_light_df[['VIOLATION DATE', 'VIOLATIONS']].groupby('VIOLATION DATE', as_index=False).sum()
    traffic_frame_sample = traffic_crash_df[['Date', 'index']].groupby('Date', as_index=False).count()

    traffic_frame_sample.columns = ["VIOLATION DATE", "Crashes"]
    red_light_frame_sample.columns = ["VIOLATION DATE", "REDLIGHT_VIOLATIONS"]
    speed_frame_sample.columns = ["VIOLATION DATE", "SPEED_VIOLATIONS"]
    res = pd.merge(traffic_frame_sample, speed_frame_sample, how='left', on='VIOLATION DATE')
    res = pd.merge(res, red_light_frame_sample, how='left', on='VIOLATION DATE')
    res.columns = ["VIOLATION DATE", "REDLIGHT_VIOLATIONS", "SPEED_VIOLATIONS", "Crashes"]
    res["Total_violations"] = res["REDLIGHT_VIOLATIONS"] + res["SPEED_VIOLATIONS"]
    res['REDLIGHT_VIOLATIONS'].fillna(0, inplace=True)
    res['SPEED_VIOLATIONS'].fillna(0, inplace=True)
    res['Crashes'].fillna(0, inplace=True)
    res['Total_violations'].fillna(0, inplace=True)

    red_crash = res[['REDLIGHT_VIOLATIONS', 'Crashes']]
    speed_crash = res[['SPEED_VIOLATIONS', 'Crashes']]

    kmeans, y_kmeans = k_means(red_crash.to_numpy())
    k_means_visualization(red_crash.to_numpy(), kmeans, y_kmeans, "Red Light Violations", "Crashes", "Date")

    kmeans, y_kmeans = k_means(speed_crash.to_numpy())
    k_means_visualization(speed_crash.to_numpy(), kmeans, y_kmeans, "Speed Light Violations", "Crashes", "Date")


def k_means(norm):
    """
    Core method that implements k-means clustering
    :param norm: dataframe to process
    :return: returns kmeans object and predicted values
    """
    kmeans = KMeans(n_clusters=3)
    y = kmeans.fit(norm)
    y_kmeans = kmeans.predict(norm)
    return kmeans, y_kmeans


def k_means_visualization(norm, kmeans, y_kmeans, x_label, y_label, by_attribute):
    """
    Method to visualize the results of k-means clustering.
    :param norm: dataframe to process
    :param kmeans: kmeans object
    :param y_kmeans: predictions object
    :param x_label: label of x-axis
    :param y_kmeans: label of y-axis
    :param by_attribute: for title
    :return: none
    """
    f1 = plt.figure()
    plt.scatter(norm[:, 0], norm[:, 1], c=y_kmeans, s=10, cmap='viridis')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("K-Means by " +  by_attribute)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50, alpha=0.5)
    plt.draw()


def heat_map(traffic_analysis, mongo_conn):
    """
    Code to retrieve data about the number of crashes in the city of Chicago with respect to the location
    :param traffic_analysis: database name
    :param mongo_conn: mongoDB connection object
    :return: None
    """
    db = mongo_conn[traffic_analysis]
    red_light_heat = list(db.violation.aggregate([
        {'$project': {
                'LONGITUDE': 1,
                'LATITUDE': 1,
                'VIOLATIONS': 1,
                'STREET_NAME': 1
        }}]))

    longitude = []
    latitude = []
    violations = []
    street_name = []
    for item in red_light_heat:
        longitude.append(item['LONGITUDE'])
        latitude.append(item['LATITUDE'])
        violations.append(item['VIOLATIONS'])
        street_name.append(item['STREET_NAME'])

    red_light_df = {}
    red_light_df['LONGITUDE'] = longitude
    red_light_df['LATITUDE'] = latitude
    red_light_df['VIOLATIONS'] = violations
    red_light_df['STREET_NAME'] = street_name

    red_light_df = pd.DataFrame(red_light_df)
    red_light_df.to_csv("red_light_heat.csv", index=False)

    speed_violations = list(db.speed.aggregate([
        {'$project': {
            'LONGITUDE': 1,
            'LATITUDE': 1,
            'VIOLATIONS': 1,
            'STREET_NAME': 1
        }}]))

    longitude = []
    latitude = []
    violations = []
    street_name = []
    for item in speed_violations:
        longitude.append(item['LONGITUDE'])
        latitude.append(item['LATITUDE'])
        violations.append(item['VIOLATIONS'])
        street_name.append(item['STREET_NAME'])

    speed_violations_df = {}
    speed_violations_df['LONGITUDE'] = longitude
    speed_violations_df['LATITUDE'] = latitude
    speed_violations_df['VIOLATIONS'] = violations
    speed_violations_df['STREET_NAME'] = street_name

    speed_violations_df = pd.DataFrame(speed_violations_df)
    speed_violations_df.to_csv("speed_violations_heat.csv", index=False)

    traffic_crashes = list(db.traffic_crash.aggregate([
                            {'$group': {'_id': {
                                        'longitude': '$LONGITUDE',
                                        'latitude': '$LATITUDE'},
                                    'violations': {'$sum': 1}}
                            }, {'$project': {
                                'longitude': 1,
                                'latitude': 1,
                                'violations': 1}}
                        ]))

    longitude = []
    latitude = []
    violations = []
    # street_name = []
    for item in traffic_crashes:
        longitude.append(item['_id']['longitude'])
        latitude.append(item['_id']['latitude'])
        violations.append(item['violations'])

    traffic_crashes_df = {}
    traffic_crashes_df['LONGITUDE'] = longitude
    traffic_crashes_df['LATITUDE'] = latitude
    traffic_crashes_df['VIOLATIONS'] = violations

    traffic_crashes_df = pd.DataFrame(traffic_crashes_df)
    traffic_crashes_df.to_csv("traffic_crashes_heat.csv", index=False)


def clustering(db_name, mongo_conn):
    db = mongo_conn[db_name]
    speed_df = pd.DataFrame(list(db.speed.find({})))
    red_light_df = pd.DataFrame(list(db.violation.find({})))
    traffic_crash_df = pd.DataFrame(list(db.traffic_crash.find({})))

    clustering_by_location(speed_df, red_light_df, traffic_crash_df)

    clustering_by_date(speed_df, red_light_df, traffic_crash_df)


if __name__ == '__main__':
    """
        Main function:
        Print the output,call the functions, prints
        the overall time taken.
    """
    start_time = time.time()

    config_path = sys.argv[1]
    mongo_connection_file = config_path + "/connection.json"
    data_file_path = config_path + "/path.json"

    global filename_dict

    filename_dict = get_files_path_params(data_file_path)

    mongo_dict = get_mongo_params(mongo_connection_file)
    db_name, mongo_conn = get_mongo_connection(mongo_dict)

    db = mongo_conn[db_name]

    time_series_analysis_combined(db_name, mongo_conn)
    time_series_analysis_separated(db_name, mongo_conn)
    time_series_analysis_red_deseasoning(db_name, mongo_conn)
    heat_map(db_name, mongo_conn)
    correlation(db_name, mongo_conn)
    clustering(db_name, mongo_conn)
    descriptive_stats(db_name, mongo_conn)
    plt.show()
