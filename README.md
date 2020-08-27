## Data Extraction and Processing
This program is to clean the data from the 3 datasets.
- Traffic_Crashes.csv
- Red_Light_Camera_Violations.csv
- Speed_Camera_Violations.csv

#### How to run the program? 
- Open the config folder.
- Set database name, hostname and port number in the connection.json file
- Set the paths for the 3 files in the path.json folder
- Open the terminal. <br />
`$ python3 DataExtraction.py <path_of_config_folder>` <br />
`$ python3 analysis.py <path of config folder>`(It should only be ran after the execution of DataExtraction.py)
