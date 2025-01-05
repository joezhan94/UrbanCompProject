import numpy as np
import os
import scipy.io

imageID = 0
for root, dirs, files in os.walk("preprocess/key_points"):
    for file in files:
        imageID = imageID + 1
        
        key_points_info = scipy.io.loadmat("preprocess/key_points/" + file)
        if_key_points = key_points_info["if_key_points"]
        all_key_points_position= key_points_info["all_key_points_position"]

        # print(if_key_points.shape)
        # print(all_key_points_position.shape)

        if_key_points = if_key_points.reshape((1,64,64))
        all_key_points_position = all_key_points_position.transpose(1,0)
        all_key_points_position = all_key_points_position.reshape((2,64,64))

        # print(if_key_points.shape)
        # print(all_key_points_position.shape)

        final_mat_savepath = "preprocess/key_points_final/" + file
        scipy.io.savemat(final_mat_savepath, mdict={'if_key_points': if_key_points, 'all_key_points_position':all_key_points_position})
        print("Image " + str(imageID) + ": Finished!")

# transform the data structure of .mat files of key points，if_key_points from (1,4096) to (1,64,64)，all_key_points_position from (4096,2) to (2,64,64)