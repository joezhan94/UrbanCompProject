# Urban Computing Course Project: City-Scale Road Lane Type Extraction from Satellite Imagery

 With the rapid pace of urban development and the pressing global
 environmental crisis, urban planners increasingly emphasize al
ternative transit methods, such as dedicated bike and bus lanes,
 when designing road networks. Automating the extraction of road
 networks from satellite imagery presented significant challenges,
 particularly in European cities with diverse transportation. This
 study addresses the challenge of extracting road networks with
 detailed lane types by a dual-task learning model, combining pixel
based multi-class semantic segmentation andgraph-basedinference
 using patch-wise road keypoint detection. Using satellite imagery
 and bike lane masks from six European cities, the model demon
strated superior accuracy (90.45%) compared to a baseline U-Net.
 The results underline the efficacy of integrating pixel-based and
 graph-based techniques in road network extraction through the
 balance of global road morphology reconstruction and local connec
tivity, overcoming limitations of previous pixel-only and graph-only
 approaches.

## Code Directory

### 1. Get satellite image and road mask data
execute get_images.py to get satellite images from Google Maps api;
execute get_masks.py to gat corresponding road network masks from OSMnx library;
manually adjust the mask images in Adobe Photoshop (or other image editing applications to make masks accurately overlay with roads in satellite images

### 2. Main model - dual-task learning
follow the readme instructions in Main_model folder

### 3. Baseline model - pixel-based U-Net 
see readme in Baseline_model folder
   
