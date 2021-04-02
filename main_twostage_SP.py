### Performing Two Stage Stochastic Programming for the Design of District Energy system ###
import os
import sys
import pandas as pd
import csv
editable_data_path =os.path.join(sys.path[0], 'EditableFile.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()

#Do we need to generate the meteorlogical data?
if editable_data['download_meto_data']=='yes':
    import donwload_windsolar_data
