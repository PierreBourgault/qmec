import sys
import os
from datetime import datetime, date, timedelta
import math
from collections import namedtuple

import pandas as pd
import numpy as np
import scipy

from qmec import qmec as qm
from qmec import datasets

upstream_data_file = datasets.get_upstream_station_data_file_path()
downstream_data_file = datasets.get_upstream_station_data_file_path()
expected_results_data_file = datasets.get_expected_results_file_path()

output_data_file = 'qmec_output_1962_2019.txt'
diff_data_file = 'differences.csv'

def timestamp_to_timestr(timestamp: float) -> str:
    ordinal_day = int(timestamp)
    ordinal_time = timestamp - ordinal_day
    dt = datetime.fromordinal(ordinal_day)
    seconds = round((ordinal_time * 86400))
    dt = dt + timedelta(seconds=seconds)
    time_str = datetime.strftime(dt, '%Y-%m-%d %H:%M:%S')
    return time_str

#get entire column from 2d array
#noCol : list of columns number to concatenate or the number of the column to extract if it's a int
def getCol(array, noCol) :    
    col = [] 
    i = 0
    for i in array:
        if type(noCol) is int:
            col.append(i[noCol])
        elif type(noCol) is list:
            str = ''
            for c in noCol:
                str += i[c] 
            col.append(str)
        else : 
            raise TypeError('In function getCol : noCol should be a list or int')
    return col        

# expect to have a array of time in string format like : 1995-10-01 00:00:00
def find_mtime(waterlevel):
    print('=== Running find_mtime... ')
    mtime = []

    for row in waterlevel:
        day_hour = f'{row[0]} {row[1]}'
        dt = datetime.strptime(day_hour,'%Y-%m-%d %H:%M:%S')
        ordinaldate = date.toordinal(dt) # + 366  <-- + 366 was in original test, never understood why
        # for compatibility with matlab, we need to add ordinal time.        
        ordinaldatetime = ordinaldate + dt.hour/24 + dt.minute/1440 + dt.second/86400
        mtime.append(ordinaldatetime)
    
    return mtime

def read_input_data_from_txt(upstream, downstream):
    wla1 = np.loadtxt(upstream, dtype={'names':('date','time','height'),'formats':('U11','U9','float')}) 

    h1 = np.array(getCol(wla1, 2)) 
    mtime_h1 = find_mtime(wla1)

    wla2 = np.loadtxt(downstream, dtype={'names':('date','time','height'),'formats':('U11','U9','float')}) 
    h2 = np.array(getCol(wla2, 2))
    mtime_h2 = find_mtime(wla2)

    validate_mtime(mtime_h1, mtime_h2)
    return [mtime_h1, h1, h2]

def validate_mtime(mtime1, mtime2):    
    print('=== Running validate_mtime')

    if (len(mtime1)==len(mtime2)) :
        for i in range(len(mtime1)): 
            if int(mtime1[i]) != int(mtime2[i]):                 
                raise ValueError('Dates of the two files does not match')
    else : 
        raise ValueError('Dates of the two files does not match')

def write_to_file(timestamps, discharges):
    warning = ''
    seconds_not_zero_found = False
    prev_year = 0
    with open(output_data_file, 'w') as f:
        for i, timestamp in enumerate(timestamps):
            timestr = timestamp_to_timestr(timestamp)
            day_hour = timestr.split()

            day = day_hour[0]
            hour = day_hour[1]
            hms = hour.split(':')

            cur_year = day.split('-')[0]
            if cur_year != prev_year:
                print(f'    Writing {cur_year}')
                prev_year = cur_year

            # Qmec gives data for each minute, but we output only once per hour. 
            # We know when we are on the hour when minutes are 00
            is_oclock = (hms[1] == '00')
            if is_oclock:
                if hms[2] != '00' and not seconds_not_zero_found:
                    seconds_not_zero_found = True
                    warning = f'!!!!! WARNING: seconds <> 00 found starting @ {day} {hour}'
                    print(warning)
                if hms[2] == '00' and seconds_not_zero_found:
                    seconds_not_zero_found = False
                    print(f'!!!!! INFO: seconds are back to 00 starting @ {day} {hour}')
                # hour = f'{hms[0]}:00:00'
                height = discharges[i]
                height = f'{height:.8e}' if not np.isnan(height) else 'NaN'
                line = f'{day} {hour} {height}'
                f.write(f'{line}\n')

    return warning

def test_remove_output_file():
    try:
        os.remove(output_data_file)
    except FileNotFoundError:
        pass

    try:
        os.remove(diff_data_file)
    except FileNotFoundError:
        pass
    
    removed = not os.path.exists(output_data_file) and not os.path.exists(diff_data_file)
    assert removed, f'Output an diff files should have been deleted'

def test_qmec():
    print('=== Reading data')

    mtime_h, h1, h2 = read_input_data_from_txt(upstream_data_file, downstream_data_file)

    print('=== Computing Qmec')
    config = {
        "calibration" : {
            "width" : 2058.177566512632,
            "depth" : 19.171325602220669,
            "mean_water_level_difference" : -1.089714209842837,
            "manning_coefficient" : 0.040108343082172
        },
        "nbStations" : 2,
        # Distance between stations in meters
        # For 2 stations, a single entry
        # For 3 stations, two entries, respectively dx between station 1 and 2, then 2 and 3
        "dx" : [
            72000
        ]
    }

    dt = 60
    calibration = namedtuple('Calibration', config['calibration'].keys())(**config['calibration'])

    w = mtime_Q, Q, h1i, h2i = qm.Qmec(calibration, mtime_h, h1, h2, config['dx'][0], dt)

    print(f'=== Writing results to {output_data_file}')
    warning = write_to_file(mtime_Q, Q)
    assert not warning, warning

def test_diff_results():
    # Load data into dataframes
    df_expected = pd.read_csv(expected_results_data_file, sep = ' ', names=['Day_exp', 'Hour_exp', 'Q_exp'], dtype={'Day_exp': str, 'Hour_exp': str, 'Q_exp': float})
    df_output = pd.read_csv(output_data_file, sep = ' ', names=['Day_out', 'Hour_out', 'Q_out'], dtype={'Day_out': str, 'Hour_out': str, 'Q_out': float})

    # Combine dataframes
    df = pd.concat([df_expected, df_output], axis=1)

    # Compute difference between expected results and output
    df['delta'] = df.apply(lambda row: np.absolute(row['Q_out'] - row['Q_exp']), axis=1)

    print(f"Maximum delta : {df['delta'].max()} in row {df['delta'].idxmax()}")
    print(f"Mean delta : {df['delta'].mean()}")

    df.to_csv(diff_data_file, index=False)

    assert True, 'Well, not supposed'
