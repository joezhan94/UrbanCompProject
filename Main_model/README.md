# Main dual-task learning model

## 1. Data preprocessing
1) in preprocess folder: put satellite images in image folder and mask images in mask folder;  
2) execute full2scribble.py to convert masks to road centerline scribbles (output in preprocess/scribble folder);  
3) execute find_key_points.py to convert scribbles to keypoint mat files (output in preprocess/key_points folder);  
4) execute format_transform.py to convert the format of keypoint mat files (output in preprocess/key_points_final folder); 
5) execute add_link.py to add link relationship to the keypoint mat files (output in preprocess/link_key_points_final folder);  
6) try with test_key_points.py and test_link.py to visualize the keypoint and link status information;  
7) execute train_test_split.py to prepare training and test data input (output in dataset)

## 2. Model training
execute train.py (training logs in logs folder, and weights stored in weights folder)  
Due to GitHub's file size limit, the trained model weights are not included in this repository. However, you can download them from https://drive.google.com/file/d/1hYVRLYmPASPLxzllsSJjw_G7XBi5nN7K/view?usp=sharing

## 3. Test
1) execute test.py to get prediction test accuract;  
2) execute test_display.py to visualize sample output versus ground truth masks
