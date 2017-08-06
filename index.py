from __future__ import print_function
import itertools
import datetime
import json
import sys
import math

from scipy.spatial import KDTree
from xml.dom import minidom
import os
import shutil
import requests
import numpy as np
import Pysolar.solar as pysolar
# import requests
import cv2
import gdal

def contour_depth(idx, hierarchy):
	if hierarchy[0][idx][2] == -1:
		return 0
	else:
		return contour_depth(hierarchy[0][idx][2], hierarchy) + 1

def poly_area(P):
	x = P[0:,0]
	y = P[0:,1]
	return np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
	
def poly_centroid(P):
	X = P[:,0]
	Y = P[:,1]
	return 1/6.0/poly_area(P) * np.asarray([\
		np.dot(X + np.roll(X, -1), np.multiply(X, np.roll(Y, -1)) - np.multiply(np.roll(X, -1), Y)),\
		np.dot(Y + np.roll(Y, -1), np.multiply(Y, np.roll(X, -1)) - np.multiply(np.roll(Y, -1), X))\
	])

def travel(origin, heading, distance): # distance in radians, heading is from true north
	lat = math.asin(math.sin(origin[1]) * math.cos(distance) + math.cos(origin[1]) * math.sin(distance) * math.cos(heading))
	return np.asarray((
		origin[0] + math.asin(math.sin(heading) * math.sin(distance) / math.cos(lat)),
		lat
	))

if __name__ == '__main__':
	HUNDRED_FEET_TO_KM = 0.3048 * 100 * 0.001

	# sys.argv expect: [__FILE__, cloudmask KML, radar tiff, metar json, elevation]
	f_cloudmask, f_radar, f_metar, f_out = sys.argv[1:] # f_elevation
	f_radar_wld = '%s.wld' % '.'.join(f_radar.split('.')[:-1])
	# need to open radar with image lib to convert to greyscale; openCV?
	radar = cv2.imread(f_radar, cv2.IMREAD_GRAYSCALE)
	
	with open(f_radar_wld, 'r') as radar_wld, open(f_metar, 'r') as metar, open(f_out, 'a') as outfile:
		# radar params
		[radar_dx, _, _, radar_dy, radar_nwx, radar_nwy] = [float(v.strip()) for v in list(radar_wld)]
		radar_nw = np.asarray([radar_nwx, radar_nwy])
		radar_dv = np.asarray([radar_dx, radar_dy])
		
		cloudmask_kml = minidom.parse(f_cloudmask)
		overlays = cloudmask_kml.getElementsByTagName('GroundOverlay')
		for overlay in overlays:
			if overlay.getElementsByTagName('name')[0].childNodes[0].nodeValue.strip() == 'Cloud Mask':
				url = overlay.getElementsByTagName('Icon')[0].getElementsByTagName('href')[0].childNodes[0].nodeValue
				
				# wld params
				latlonbox = overlay.getElementsByTagName('LatLonBox')[0]
				bounds = [
					float(latlonbox.getElementsByTagName('north')[0].childNodes[0].nodeValue),
					float(latlonbox.getElementsByTagName('south')[0].childNodes[0].nodeValue),
					float(latlonbox.getElementsByTagName('east')[0].childNodes[0].nodeValue),
					float(latlonbox.getElementsByTagName('west')[0].childNodes[0].nodeValue)
				]
				
				cloudmask_response = requests.get(url, stream=True)
				d, f = url.split('/')[-2:]
				if not os.path.exists('/tmp/cmasks/%s' % d):
					os.makedirs('/tmp/cmasks/%s' % d)
					
				with open('/tmp/cmasks/%s/%s' % (d, f), 'wb') as tmpfile:
					cloudmask_response.raw.decode_content = True
					shutil.copyfileobj(cloudmask_response.raw, tmpfile)
				
				cloudmask = cv2.imread('/tmp/cmasks/%s/%s' % (d, f), cv2.IMREAD_GRAYSCALE)
				cloudmask_nw = np.asarray([ bounds[3], bounds[0] ])
				cloudmask_dv = np.divide(
					np.asarray([bounds[2] - bounds[3], bounds[1] - bounds[0]]),
					np.asarray(cloudmask.shape)
				)
				break
		
		# cloudmask_hdf = gdal.Open(f_cloudmask)
		# cloudmask = gdal.Open(cloudmask_hdf.GetSubDatasets()[30][0]).ReadAsArray()
		# lon = gdal.Open(cloudmask_hdf.GetSubDatasets()[6][0]).ReadAsArray()
		# lat = gdal.Open(cloudmask_hdf.GetSubDatasets()[5][0]).ReadAsArray()
		# cloudmask_lookup = []
		# cloudmask_coords = []
		
		# for i in range(cloudmask.shape[0]):
		# 	for j in range(cloudmask.shape[1]):
		# 		cloudmask_lookup.append(cloudmask[i, j])
		# 		cloudmask_coords.append([ lon[i, j], lat[i, j] ])
		
		# cloudmask_kdtree = KDTree(cloudmask_coords)
		
		# cardinal_directions = itertools.permutations([-1, 0, 1], 2)
		# for i, row in enumerate(border_mask):
		# 	for j, pix in enumerate(row):
		# 		if pix > 0:
		# 			for direction in cardinal_directions:
		# 				if globe_mask[i + direction[0], j + direction[1]] > 0 and not border_mask[i + direction[0], j + direction[1]]:
		# 					cloudmask[i, j] = cloudmask[i + direction[0], j + direction[1]]
		
		# cloudmask = np.logical_and(
		# 	globe_mask > 0,
		# 	np.logical_or.reduce((
		# 		cv2.inRange(cloudmask, (0, 0, 255), (0, 0, 255)), # ice cloud
		# 		cv2.inRange(cloudmask, (0, 255, 255), (0, 255, 255)), # ice cloud weak
		# 		cv2.inRange(cloudmask, (255, 255, 0), (255, 255, 0)), # supercooled water cloud
		# 		cv2.inRange(cloudmask, (255, 0, 0), (255, 0, 0)) # water cloud
		# 	))
		# )
		
		metar_geojson = json.load(metar)
		metar_lookup = [feature['properties'] for feature in metar_geojson['features']]
		metar_kd = KDTree([feature['geometry']['coordinates'] for feature in metar_geojson['features']])
		
		# _, cloud_contours, cloud_hierarchy = cv2.findContours(cloudmask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		# cloud_polies = [Polygon([deprojector.deproject(pt[0]) for pt in poly]) for poly in cloud_contours if poly.shape[0] > 2]
		
		# add visilibity? Eh, maybe not, might be cheaper to still do the point-in-polygon thing anyways since it's per-point
		# note: we're also considering inner contours too. Not likely to have big holes, but then it won't be expensive anyways
		_, radar_contours, _ = cv2.findContours((radar > 0).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# TODO reproject radar contour to latlon
		# actually, no need: the images are already WGS84
		# projection is simply taking into account the world file
		# lon: nwx + x * dx, lat: nwy + y * dy
		
		# oh yeah fill that array, that's the spot
		sky_points = set()
		contours_skipped = 0
		for contour_idx, px_contour in enumerate(radar_contours):
			if px_contour.shape[0] < 3:
				contours_skipped += 1
				continue
				
			print('\r%d / %d contours completed, %d skipped, %d sky pixels hit' % (contour_idx - 1, len(radar_contours), contours_skipped, len(sky_points)), end='', file=sys.stderr)
			if len(px_contour) > 2:
				# where all the magic happens
				contour = np.multiply(np.squeeze(px_contour).astype(np.float32), radar_dv) + radar_nw
				ceil = None
				# visited = set() # eh, not worth it.
				for num_stations_exponent in itertools.count(1):
					if 2 ** num_stations_exponent > len(metar_lookup):
						raise 'Either the data format to the METAR station has changed, the power is out across the CONUS, or it\'s a perfectly sunny day. Either way, go outside and get yourself some ice cream!'
					# consider approximate for faster results - need characteristic separation between stations in degrees
					centroid = poly_centroid(contour)
					if not np.any(np.logical_or(np.isinf(centroid), np.isnan(centroid))):
						distances, stations = metar_kd.query(np.multiply(np.asarray([ 1 -1 ]), centroid), 2 ** num_stations_exponent)
						for station in stations:
							if 'ceil' in metar_lookup[station]:
								ceil = metar_lookup[station]['ceil']
								break
					else:
						print('Warning: could not calculate valid centroid for radar blob with contour idx %d.' % contour_idx)
						break
						
					if ceil != None:
						break
				
				if ceil == None:
					continue
				# need contour data, else assume the land under the cloud is totally flat
				
				for P in contour:
					# TODO implement class ElevMap { at(LonLat): float }
					# elev_map.at(P) # although maybe we assume the elev is pretty similar
					# TODO: get T := DateTime, as mean time for all the data imputs
					# use small-angle approximation so that the pythag output is still radians
					T = datetime.datetime(2017, 07, 27, 22, 15)
					sky_intercept = travel(
P * math.pi / 180,
math.pi - (pysolar.GetAzimuth(P[1], P[0], T) * math.pi / 180), # pysolar uses latlon; we use lonlat
ceil * HUNDRED_FEET_TO_KM / math.tan(pysolar.GetAltitude(P[1], P[0], T) * math.pi / 180) / 6370 # TEMP: earth radius will find a better home later
)
					cloudmask_coords = np.divide(sky_intercept * 180 / math.pi - cloudmask_nw, cloudmask_dv).astype(np.uint16)
					cloudmask_px = cloudmask[cloudmask_coords[1], cloudmask_coords[0]]
					
					if cloudmask_px == 0 or \
						cloudmask_px == 42:
					   outfile.write('[%s, %s],\n' % tuple(P))
					   sky_points.add((int(round(cloudmask_coords[1])), int(round(cloudmask_coords[0]))))
					
					# clear_flag = True
					# for idx, cloud_poly in enumerate(cloud_polies):
					# 	if cloud_poly.contains(Point(sky_intercept)):
					# 		if contour_depth(idx, cloud_hierarchy) % 2 == 0: # it's a cloud by parity
					# 			clear_flag = False
					# 		break
							
					# if clear_flag:
					# 	# rainbow_points.append(P)
					# 	outfile.write('[%s, %s],\n' % tuple(P))
