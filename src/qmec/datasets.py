from importlib import resources

def get_upstream_station_data_file_path():
    with resources.path('qmec.data', 'h_3280_Neuville_1962_2019.txt') as f:
        return f

def get_downstream_station_data_file_path():
    with resources.path('qmec.data', 'h_3100_StFrancoisIO_1962_2019.txt') as f:
        return f

def get_expected_results_file_path():
    with resources.path('qmec.data', 'qmec_expected_1962_2019.txt') as f:
        return f
