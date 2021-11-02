import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import os
from tsfresh import extract_features

def get_datum(json_filename, incident_number):

    df_categorised = pd.read_json(json_filename)
    linspace_1 = np.linspace(-6, 2.875, 72)

    # Changed zoomed_in_df to include t=0 and the other variables
    if str(json_filename) == 'data/uncategorised.json' or json_filename == 'data/categorised.json':

        # Get status of incident
        status = df_categorised['status']
        status = status[incident_number]
        # Get detailed info
        accel_df = df_categorised['detail']
        detail_dict = accel_df[incident_number]
        # Less detailed data
        gps_df = df_categorised['journey']
        journey_dict = gps_df[incident_number]

        # Zoomed out data, has long term speed and accelerometer data
        zoomed_out_df = pd.DataFrame.from_dict(journey_dict)
        # Zoomed in data, use to get the speed in smaller interval
        zoomed_in_df = pd.DataFrame.from_dict(detail_dict)

        # Extract the higher resolution accelerometer data for the crash
        forces = zoomed_in_df['forces']
        zoomed_in_tilts = []
        for second in forces:
            force_1 = pd.DataFrame.from_dict(second)
            zoomed_in_tilts.append(force_1)

        zoomed_in_tilts = pd.concat(zoomed_in_tilts)
        zoomed_in_tilts = zoomed_in_tilts.rename(columns={'index':'timeoffset'})

        zoomed_in_tilts['timeoffset'] = linspace_1

        zoomed_in_df = zoomed_out_df[zoomed_out_df['event'] == 'CDistance']
        alert_row = zoomed_out_df[zoomed_out_df['event'] == 'Alert']
        zoomed_in_df = zoomed_in_df.append(alert_row)
        zoomed_in_df = zoomed_in_df.sort_index()

        return [zoomed_in_df, zoomed_in_tilts, zoomed_out_df, status]

    elif str(json_filename) == 'data/unlinked.json':

        zoomed_in_df = []
        accel_df = df_categorised['detail']
        detail_dict = accel_df[incident_number]
        # Zoomed in data, use to get the speed in smaller interval
        zoomed_in_df = pd.DataFrame.from_dict(detail_dict)

        zoomed_in_df = zoomed_in_df.sort_values(by=['timeoffset'])

        forces = zoomed_in_df['forces']
        zoomed_in_tilts = []
        for second in forces:
            force_1 = pd.DataFrame.from_dict(second)
            zoomed_in_tilts.append(force_1)

        zoomed_in_tilts = pd.concat(zoomed_in_tilts)
        zoomed_in_tilts = zoomed_in_tilts.rename(columns={'index':'timeoffset'})

        zoomed_in_tilts['timeoffset'] = linspace_1

        return [zoomed_in_df, zoomed_in_tilts]


def get_data(json_filename):
    df = pd.read_json(json_filename)
    all_data = []
    for i in tqdm(range(len(df))):
        data = get_datum(json_filename, i)
        all_data.append(data)

    return all_data

def get_timeseries(tilts):
    tilts = np.array(tilts)
    tilts = tilts[:,1:]
    tilt_timeseries = tilts.transpose()
    tilt_timeseries = tilt_timeseries.astype('float')
    tilt_timeseries = np.reshape(tilt_timeseries,(3,72))

    return tilt_timeseries

def get_tilt_timeseries(data):
    # Extracts timeseries of tilts for whole dataset (output shape = (incidents, len_ts, axes))
    X = np.empty((len(data),72,3))
    for i in range(len(data)):
        incident = data[i]
        tilts = np.array(incident[1])
        tilts = tilts[:,1:]
        X[i,:,:] = tilts
    X = np.reshape(X,(len(data),72,3))
    return X

def get_labels(cat_data):
    y_test = []
    for i in range(len(cat_data)):
        incident = cat_data[i]
        if 'Correct' in incident[3]:
            y_test.append(1)
        else:
            y_test.append(0)

    return y_test

def get_mags(X):
    mag_X = np.empty((len(X), 72))
    for i in range(len(X)):
        for k in range(72):
            vector = np.array(X[i,k,:])
            vector = [float(i) for i in vector]
            mag = np.linalg.norm(vector[:])
            mag_X[i,k] = mag
    return mag_X

def tsfresh_extraction(data):
    # Get tilts, calibrate and get mags
    tilts = get_tilt_timeseries(data)
    tilts = calibrate_remove_z(tilts)
    mags = get_mags(tilts)
    mags = mags.flatten()

    # Get repeating time array
    time = np.linspace(-6, 2.875, 72)
    times = []
    for i in range(len(data)):
        times.append(time)
    time = np.array(times)
    time = time.flatten()

    # Get ids
    ids = []
    for i in range(len(data)):
        for k in range(72):
            ids.append(i)
    ids = np.array(ids)

    # Collect into df
    ts_data = [ids, time, mags]
    ts_data = np.transpose(np.array(ts_data))
    ts_df = pd.DataFrame(ts_data)
    ts_df.columns = ['id', 'time', 'mag']

    features = extract_features(ts_df, column_id="id", column_sort="time", column_kind=None, column_value=None)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.dropna(axis=1)

    print(features.shape)

    return features

###########################################################
### Calibration Functions                               ###
### Used to calibrate tilt data + adjust for gravity    ###
###########################################################
def calibrate_remove_z(data):
    cali_data = np.empty((int(data.shape[0]), int(data.shape[1]), 2))
    z_dirs = []
    for i in range(len(data)):
        # Calculate the average over the first 4 seconds (average shouldn't be affected by the crash)
        x_av = np.sum(data[i,:32,0])/len(data[i,:32,0])
        y_av = np.sum(data[i,:32,1])/len(data[i,:32,1])
        z_av = np.sum(data[i,:32,2])/len(data[i,:32,2])

        max_av = np.argmax(np.abs([x_av, y_av, z_av])) # find the direction with largest absolute value (should be z)
        z_dirs.append(max_av)

        if max_av == 0:
            cali_data[i,:,0] = data[i,:,1] - y_av
            cali_data[i,:,1] = data[i,:,2] - z_av
        if max_av == 1:
            cali_data[i,:,0] = data[i,:,0] - x_av
            cali_data[i,:,1] = data[i,:,2] - z_av
        if max_av == 2:
            cali_data[i,:,0] = data[i,:,0] - x_av
            cali_data[i,:,1] = data[i,:,1] - y_av

    return cali_data

def calibrate_tilts(data):
    cali_data = np.empty(data.shape)
    for i in range(len(data)):
        # Calculate the average over the first 4 seconds (average shouldn't be affected by the crash)
        x_av = np.sum(data[i,:32,0])/len(data[i,:32,0])
        y_av = np.sum(data[i,:32,1])/len(data[i,:32,1])
        z_av = np.sum(data[i,:32,2])/len(data[i,:32,2])
        # Remove that average to centre the data
        cali_data[i,:,0] = data[i,:,0] - x_av
        cali_data[i,:,1] = data[i,:,1] - y_av
        cali_data[i,:,2] = data[i,:,2] - z_av
    return cali_data

###########################################################
### PICKLE FUNCTIONS                                    ###
### Used to save and load python data structs to file   ###
###########################################################
def save_list(list_to_save, folder_name, file_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        print('Making Directory:' + str(folder_name))
    with open(folder_name + '/' + file_name, 'wb') as fp:
        pickle.dump(list_to_save, fp)

def load_list(folder_name, file_name):
    with open(folder_name + '/' + file_name, 'rb') as fp:
        return pickle.load(fp)
###########################################################
### PLOTTING                                            ###
### Functions needed to plot example incident           ###
###########################################################
def plot_tilts_zi(df, title):
    tilts = get_timeseries(df)
    timeoffset = np.linspace(-6, 2.875, 72)
    plt.plot(timeoffset, tilts[0,:], label='tiltx', color='red')
    plt.plot(timeoffset, tilts[1,:], label='tilty', color='blue')
    plt.plot(timeoffset, tilts[2,:], label='tiltz', color='green')
    plt.title(title), plt.xlabel('TimeOffset'), plt.ylabel('Tilt')
    plt.legend(loc='best')
    #plt.show()

def plot_tilts_zo(df, title):
    plt.plot(df['timeoffset'], df['tiltx'], label='tiltx', color='red')
    plt.plot(df['timeoffset'], df['tilty'], label='tilty', color='blue')
    plt.plot(df['timeoffset'], df['tiltz'], label='tiltz', color='green')
    plt.title(title), plt.xlabel('TimeOffset'), plt.ylabel('Tilt')
    plt.legend(loc='best')
    #plt.show()

def plot_speeds(df, title):
    plt.plot(df['timeoffset'], df['speed'], label='Speed', color='blue')
    plt.title(title), plt.xlabel('TimeOffset'), plt.ylabel('Speed')
    #plt.show()

def plot_grid(df, title):
    X = df['gridx']
    Y = df['gridy']
    plt.plot(X, Y, label='Path', color='blue')
    plt.scatter(X.iloc[0], Y.iloc[0], label='Start', marker='*', color='red')
    plt.scatter(X.iloc[-1], Y.iloc[-1], label='End', marker='o', color='red')
    plt.scatter(0,0, label='Incident', color='red', marker='x')
    plt.legend(loc='best')
    plt.title(title), plt.xlabel('gridX'), plt.ylabel('gridY')
    #plt.show()

def plot_example(incident, title):
    fig = plt.figure(figsize=(14,10))
    fig.suptitle(title, fontsize=16)
    plt.subplot(2,3,1)
    plot_tilts_zi(incident[1], 'Zoomed In Tilts')
    plt.subplot(2,3,2)
    plot_tilts_zo(incident[2], 'Zoomed Out Tilts')
    plt.subplot(2,3,3)
    plot_speeds(incident[0], 'Zoomed In Speeds')
    plt.subplot(2,3,4)
    plot_speeds(incident[2], 'Zoomed Out Speeds')
    plt.subplot(2,3,5)
    plot_grid(incident[2], 'Zoomed Out Locations')
    plt.subplot(2,3,6)
    plot_grid(incident[0], 'Zoomed In Locations')
    plt.show()