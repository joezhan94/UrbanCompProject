import os
import math
import requests
from PIL import Image
from pathlib import Path


# Constants
ZOOM = 19
TILE_SIZE = 4096
SUB_TILE_SIZE = 576
ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0

# Define the region map
region_map = {
    "amsterdam": [52.376805, 4.882880, 52.363139, 4.904190],
    "london": [51.510040, -0.027977, 51.494132, -0.008300],
    "utrecht": [52.0, 4.9, 52.2, 5.1],
    "copenhagen": [55.6, 12.4, 55.8, 12.7],
    "strasbourg": [48.5, 7.6, 48.7, 7.8],
    "paris": [48.8, 2.2, 49.0, 2.5],
}


def get_regions():
    regions = []
    for name, array in region_map.items():
        center_gps = ((array[1] + array[3]) / 2, (array[0] + array[2]) / 2)
        radius_x = math.ceil(abs(array[3] - array[1]) * 810 * 2 / 576)
        radius_y = math.ceil(abs(array[2] - array[0]) * 500 * 2 / 576)
        # print(radius_x, radius_y)
        # if name in ["denver", "kansas city", "san diego", "pittsburgh", "montreal", "vancouver", "tokyo", "saltlakecity", "paris", "amsterdam"]:
        #     radius_x = 1
        #     radius_y = 1
        regions.append((name, center_gps, radius_x, radius_y))
    return regions



def get_satellite_image(center_gps, zoom, api_key):
    request = "https://maps.googleapis.com/maps/api/staticmap?center="+str(center_gps[1])+","+str(center_gps[0])+"&zoom="+str(zoom)+"&key="+api_key+"&size=576x576&maptype=satellite"
    # print(request)
    response = requests.get(request)
    f = open("data/imagery/test.png", "wb")
    f.write(response.content)
    f.close()
    image = Image.open("data/imagery/test.png")
    return image

def main(api_key, out_dir):
    regions = get_regions()
    print(regions)
    SIZE = 10
    s = 0 - (SIZE // 2)
    l = (SIZE // 2) + 1
    boundsDict = {}
    for region in regions:
        largerimage = Image.new('RGB', (576*(SIZE+1), 576*(SIZE+1)))
        print(largerimage.size)
        minX, minY = math.inf, math.inf
        maxX, maxY = 0, 0
        for x in range(s, l):
            for y in range(s, l):
                center_gps = (region[1][0] + x * 810.0 / float(2 ** ZOOM), region[1][1] + y * 500.0 / float(2 ** ZOOM))
                # Print bounding box
                bounding = ((center_gps[1] - 405 / float(2 ** ZOOM)), (center_gps[0] - 288 / float(2 ** ZOOM)), (center_gps[1] + 405 / float(2 ** ZOOM)), (center_gps[0] + 288 / float(2 ** ZOOM)))
                # print(center_gps)
                print(x, y)
                if bounding[0] < minX:
                    minX = bounding[0]
                if bounding[1] < minY:
                    minY = bounding[1]
                if bounding[2] > maxX:
                    maxX = bounding[2]
                if bounding[3] > maxY:
                    maxY = bounding[3]
                image = get_satellite_image(center_gps, ZOOM, api_key)
                largerimage.paste(image, ((x-s)*576, (-y-s)*576))
        largerimage.save(os.path.join(out_dir, f"{region[0]}.png"))
        boundsDict[region[0]] = (minX, minY, maxX, maxY)
    print(boundsDict)
if __name__ == "__main__":
    import sys
    api_key = "AIzaSyBFNqctWiR9_PPaVxHx33znfwp1F9h4Qb4"
    out_dir = "data/imagery"
    main(api_key, out_dir)