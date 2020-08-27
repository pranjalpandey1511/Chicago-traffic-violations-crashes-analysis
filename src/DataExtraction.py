__author__ = "Abhay Rajendra Dixit "
__author__ = "Pranjal Pandey"
__author__ = "Ravikiran Jois Yedur Prabhakar"

import datetime
import json
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from pymongo import MongoClient


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


def read_data():
    """
         This function reads the source file, selects the important attribute
         and convert into the data-set to data frame using pandas library.
         :return red_light_df: Data-frame obtained from red-light violations
         :return crash_df: Data-frame obtained from traffic crashes violations
         :return speed_df: Data-frame obtained from speed-camera violations
    """
    red_light_df = pd.read_csv(filename_dict["redlight_data_path"], usecols=["ADDRESS", "VIOLATION DATE", "VIOLATIONS",
                                                                             "LATITUDE", "LONGITUDE"])

    crash_df = pd.read_csv(filename_dict["traffic_crash_data_path"],
                           usecols=["CRASH_DATE", "STREET_NO", "STREET_NAME", "STREET_DIRECTION", "POSTED_SPEED_LIMIT",
                                    "FIRST_CRASH_TYPE", "TRAFFICWAY_TYPE", "PRIM_CONTRIBUTORY_CAUSE", "LATITUDE",
                                    "LONGITUDE"])

    speed_df = pd.read_csv(filename_dict["speed_violations_data_path"],
                           usecols=["ADDRESS", "VIOLATION DATE", "VIOLATIONS", "LATITUDE", "LONGITUDE"])

    crash_df[['Date', 'Time', 'M']] = crash_df.CRASH_DATE.str.split(" ", expand=True, )

    return red_light_df, crash_df, speed_df


def process_red_light_data(red_light_frame):
    """
         This function reads and process red-light violations data-frame and
         perform transformations to clean and prepare the red light violations
         data-set.
         :param red_light_frame: Red light violations data-frame with selected attributes
    """
    print("Processing red light violations dataset...")
    red_light_frame["ADDRESS"].replace(to_replace=[" roa$| ROA$", " ave$| AVE$", " stree$| STREE$",
                                                   " boulev$| BOULEV$", " dr$| DR$", " parkwa$| PARKWA$", " st$| ST$"],
                                       value=[" ROAD", " AVENUE", " STREET", " BOULEVARD", " DRIVE", " PARKWAY", " STREET"],
                                       regex=True, inplace=True)

    red_light_frame["STREET_NO_DIR"] = red_light_frame["ADDRESS"].str.split(' ', 2)
    red_light_frame["STREET_NAME"] = red_light_frame["ADDRESS"].str.split().str[2:]
    red_light_frame["STREET_NAME"] = [' '.join(map(str, l)) for l in red_light_frame['STREET_NAME']]
    red_light_frame[['STREET_NO', 'STREET_DIR', 'OTHERS']] = pd.DataFrame(red_light_frame.STREET_NO_DIR.tolist(),
                                                                          index=red_light_frame.index)
    del red_light_frame['OTHERS']


def process_speed_data(speed_sample):
    """
         This function reads and process speed-camera violations data-frame and
         perform transformations to clean and prepare the speed camera violations
         data-set.
         :param speed_sample: Speed-camera violations data-frame with selected attributes
    """
    print("Processing speed camera violations dataset...")
    speed_sample["ADDRESS"].replace(to_replace=[" rd$| RD$", " av$| AV$| ave$| AVE$", " st$| stree$| STREE$| ST$",
                                                " blvd$| BLVD$", " dr$| DR$", " parkwa$| PARKWA$", " hwy$| HWY$"],
                                    value=[" ROAD", " AVENUE", " STREET", " BOULEVARD", " DRIVE", " PARKWAY",
                                           " HIGHWAY"],
                                    regex=True, inplace=True)

    speed_sample["STREET_NO_DIR"] = speed_sample["ADDRESS"].str.split(' ', 2)
    speed_sample["STREET_NAME"] = speed_sample["ADDRESS"].str.split().str[2:]
    speed_sample["STREET_NAME"] = [' '.join(map(str, l)) for l in speed_sample['STREET_NAME']]
    speed_sample[['STREET_NO', 'STREET_DIR', 'OTHERS']] = pd.DataFrame(speed_sample.STREET_NO_DIR.tolist(),
                                                                       index=speed_sample.index)

    del speed_sample['OTHERS']


def process_crash_data(traffic_frame):
    """
         This function reads and process traffic crashes data-frame and
         perform transformations to clean and prepare the traffic crashes
         data-set.
         :param traffic_frame: Traffic crashes data-frame with selected attributes
    """
    print("Processing traffic crashes dataset...")
    traffic_frame["STREET_NAME"].replace(to_replace=[" rd$| RD$", " ave$| AVE$| av$| AV$", " st$| ST$",
                                                     " blvd$| BLVD$", " dr$| DR$", " pkwy$| PKWY$"],
                                         value=[" ROAD", " AVENUE", " STREET", " BOULEVARD", " DRIVE", " PARKWAY"],
                                         regex=True, inplace=True)

    traffic_frame['STREET_NAME'].replace(np.nan, -9999, inplace=True)


def process_data(traffic_frame, red_light_frame, speed_sample):
    """
         This function reads and process data frames by passing it to different
         functions which performs various transformations to clean and prepare
         the data-sets.
         :param traffic_frame: Traffic crashes data-frame with selected attributes
         :param red_light_frame: Red light violations data-frame with selected attributes
         :param speed_sample: Speed-camera violations data-frame with selected attributes
         :return clean_red_light_frame: Transformed cleaned data-frame of red light violations
         :return clean_traffic_frame: Transformed cleaned data-frame of traffic crashes
         :return clean_speed_frame:  Transformed cleaned data-frame of speed camera violations
    """
    # For Red Light Violations
    process_red_light_data(red_light_frame)

    # For Traffic Crashes
    process_crash_data(traffic_frame)

    # For Speed Camera Violations
    process_speed_data(speed_sample)

    clean_red_light_frame, clean_traffic_frame, clean_speed_frame = select_attributes(red_light_frame, traffic_frame,
                                                                                      speed_sample)
    return clean_red_light_frame, clean_traffic_frame, clean_speed_frame


def select_attributes(red_light_frame, traffic_frame, speed_frame):
    """
         This function drops the irrelevant attributes from the data-frame.
         :param traffic_frame: Traffic crashes data-frame with selected attributes
         :param red_light_frame: Red light violations data-frame with selected attributes
         :param speed_frame: Speed-camera violations data-frame with selected attributes
         :return red_light_frame: Transformed data-frame of red light violations
         :return traffic_frame: Transformed  data-frame of traffic crashes
         :return speed_frame: Transformed data-frame of speed camera violations
    """
    traffic_frame.drop(["CRASH_DATE", "Time", "M"], axis=1, inplace=True)
    red_light_frame.drop(["STREET_NO_DIR", "ADDRESS"], axis=1, inplace=True)
    speed_frame.drop(["STREET_NO_DIR", "ADDRESS"], axis=1, inplace=True)
    return red_light_frame, traffic_frame, speed_frame


def insert_data_to_mongo(traffic_analysis, mongo_con, traffic_frame, red_light_frame, speed_frame):
    """
        This function loads the clean data to mongo db.
        :param traffic_analysis: Name of the database
        :param mongo_con: Contains the connection parameters of mongodb
        :param red_light_frame: Transformed and cleaned data-frame of red-light violations
        :param traffic_frame: Transformed and cleaned data-frame of traffic-crashes
        :param speed_frame: Transformed and cleaned data-frame of speed-camera violations
    """
    db = mongo_con[traffic_analysis]
    traffic_crash_collection = db['traffic_crash']
    violation_collection = db['violation']
    speed_camera_collection = db['speed']

    traffic_frame["Date"].replace({" ": ""}, inplace=True)
    traffic_frame["Date"] = pd.to_datetime(traffic_frame["Date"])

    traffic_frame.reset_index(inplace=True)
    traffic_frame_dict = traffic_frame.to_dict("records")

    print("Inserting traffic data to MongoDB")
    traffic_crash_collection.insert_many(traffic_frame_dict)

    print("Truncating documents before 2015")
    traffic_crash_collection.delete_many(
        {
            'Date': {
                '$lt': datetime.datetime(2015, 1, 1, 0, 0, 0, 0)
            }
        }
    )

    print("Insert Red Light Data to traffic collection")
    red_light_frame["VIOLATION DATE"].replace({" ": ""}, inplace=True)
    red_light_frame["VIOLATION DATE"] = pd.to_datetime(red_light_frame["VIOLATION DATE"])
    red_light_frame_dict = red_light_frame.to_dict("records")
    violation_collection.insert_many(red_light_frame_dict)

    violation_collection.delete_many(
        {
            'VIOLATION DATE': {
                '$lt': datetime.datetime(2015, 1, 1, 0, 0, 0, 0)
            }
        }
    )

    print("Insert Speed Camera Data to traffic collection")
    speed_frame["VIOLATION DATE"].replace({" ": ""}, inplace=True)
    speed_frame["VIOLATION DATE"] = pd.to_datetime(speed_frame["VIOLATION DATE"])
    speed_frame_dict = speed_frame.to_dict("records")
    speed_camera_collection.insert_many(speed_frame_dict)

    speed_camera_collection.delete_many(
        {
            'VIOLATION DATE': {
                '$lt': datetime.datetime(2015, 1, 1, 0, 0, 0, 0)
            }
        }
    )


def get_stats(red_light_frame, speed_frame, traffic_crash):
    """
        This function provides the descrptive statistics for the numeric
        fields in the dataset
        :param red_light_frame: Transformed and cleaned data-frame of red-light violations
        :param traffic_crash: Transformed and cleaned data-frame of traffic-crashes
        :param speed_frame: Transformed and cleaned data-frame of speed-camera violations
    """
    red_light_stats = red_light_frame.VIOLATIONS.describe()
    speed_camera_stats = speed_frame.VIOLATIONS.describe()
    traffic_crash = traffic_crash.POSTED_SPEED_LIMIT.describe()
    print("Descriptive statistics for the number of Red Light Violations")
    print(red_light_stats)
    print()
    print("Descriptive statistics for the number of Speed Camera Violations")
    print(speed_camera_stats)
    print()
    print("Descriptive statistics for the number of Posted Speed Limit")
    print(traffic_crash)


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
    red_light_frame, traffic_frame, speed_frame = read_data()
    red_light_frame, traffic_frame, speed_frame = process_data(traffic_frame, red_light_frame, speed_frame)

    insert_data_to_mongo(db_name, mongo_conn, traffic_frame, red_light_frame, speed_frame)
    print()
    print("--------------- Data Inserted to MongoDB ----------------")
    print()
    get_stats(red_light_frame, speed_frame, traffic_frame)

    print()

    print("Total time taken for the process: ", time.time() - start_time)
