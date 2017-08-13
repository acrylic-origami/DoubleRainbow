from __future__ import print_function
import itertools
import datetime
import json
import sys
import math

from scipy.spatial import KDTree
import scipy.misc
from xml.dom import minidom
from HTMLParser import HTMLParser
import datetime
import re
import os
import shutil
import requests
import numpy as np
import Pysolar.solar as pysolar
# import requests
import cv2
# import gdal

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

class CloudmaskDirectoryParser(HTMLParser):
	def __init__(self):
		HTMLParser.__init__(self)
		self.latest = None
	def datetime_from_filename(self, url):
		[_, _, year, day, time] = url.split('/')[-1][:-len('.kml')].split('_')
		return datetime.datetime.strptime('%d %s %s' % (int(year), day, time), '%Y %j %H%M')
	def handle_starttag(self, tag, attrs):
		if tag.lower() == 'a':
			href = [attr[1] for attr in attrs if attr[0] == 'href'][0]
			if re.match('clavrx_goes13_\d+_\d+_\d+\.kml', href):
				date = self.datetime_from_filename(href)
				
				if self.latest == None or date > self.latest:
					self.latest = date

if __name__ == '__main__':
	HUNDRED_FEET_TO_KM = 0.3048 * 100 * 0.001
	
	url_cloudmask_directory = 'http://cimss.ssec.wisc.edu/clavrx/google_earth/goes_east_kml/'
	
	res_cloudmask_directory = requests.get(url_cloudmask_directory)
	cloudmask_directory_parser = CloudmaskDirectoryParser()
	cloudmask_directory_parser.feed(res_cloudmask_directory.text)
	now = cloudmask_directory_parser.latest
	
	url_cloudmask = 'http://cimss.ssec.wisc.edu/clavrx/google_earth/goes_east_kml/clavrx_goes13_%s.kml' % now.strftime('%Y_%j_%H%M')
	
	url_radar = 'http://mesonet.agron.iastate.edu/archive/data/%s/GIS/uscomp/n0r_%s.png' % (now.strftime('%Y/%m/%d'), now.strftime('%Y%m%d%H%M'))
	url_radar_wld = 'http://mesonet.agron.iastate.edu/archive/data/%s/GIS/uscomp/n0r_%s.wld' % (now.strftime('%Y/%m/%d'), now.strftime('%Y%m%d%H%M'))
	url_metar = 'http://aviationweather.ncep.noaa.gov/gis/scripts/MetarJSON.php?date=%s' % now.strftime('%Y%m%d%H%M')
	
	res_cloudmask = requests.get(url_cloudmask, stream=True)
	res_radar = requests.get(url_radar, stream=True)
	res_radar_wld = requests.get(url_radar_wld, stream=True)
	res_metar = requests.get(url_metar, stream=True)
	
	tmp_dir = '/tmp/%s' % now.strftime('%Y%m%d%H%M')
	
	if not os.path.exists(tmp_dir):
		os.makedirs(tmp_dir)
		
	with open('%s/radar.png' % tmp_dir, 'wb') as f:
		res_radar.raw.decode_content = True
		shutil.copyfileobj(res_radar.raw, f)
	
	radar = cv2.imread('%s/radar.png' % tmp_dir, cv2.IMREAD_COLOR)
	radar_gray = cv2.cvtColor(radar, cv2.COLOR_BGR2GRAY)
	radar_mask = radar_gray > 0
	
	cloudmask_kml = minidom.parseString(res_cloudmask.text)
	overlays = cloudmask_kml.getElementsByTagName('GroundOverlay')
	for overlay in overlays:
		if overlay.getElementsByTagName('name')[0].childNodes[0].nodeValue.strip() == 'Cloud Mask':
			url = overlay.getElementsByTagName('Icon')[0].getElementsByTagName('href')[0].childNodes[0].nodeValue
			
			# wld params
			latlonbox = overlay.getElementsByTagName('LatLonBox')[0]
			cloudmask_bounds = [
				float(latlonbox.getElementsByTagName('north')[0].childNodes[0].nodeValue),
				float(latlonbox.getElementsByTagName('south')[0].childNodes[0].nodeValue),
				float(latlonbox.getElementsByTagName('east')[0].childNodes[0].nodeValue),
				float(latlonbox.getElementsByTagName('west')[0].childNodes[0].nodeValue)
			]
			
			cloudmask_response = requests.get(url, stream=True)
				
			with open('%s/cloudmask.png' % tmp_dir, 'wb') as tmpfile:
				cloudmask_response.raw.decode_content = True
				shutil.copyfileobj(cloudmask_response.raw, tmpfile)
			
			cloudmask = cv2.imread('%s/cloudmask.png' % tmp_dir, cv2.IMREAD_GRAYSCALE)
			cloudmask_nw = np.asarray([ cloudmask_bounds[3], cloudmask_bounds[0] ])
			cloudmask_dv = np.divide(
				np.asarray([cloudmask_bounds[2] - cloudmask_bounds[3], cloudmask_bounds[1] - cloudmask_bounds[0]]),
				np.asarray(cloudmask.shape)
			)
			break
	
	radar_wld = res_radar_wld.text
	metar_json = res_metar.json()
	
	base_out_dir = sys.argv[1]
	out_dir = '%s/%s/%s/' % (base_out_dir, now.strftime('%Y%m%d'), now.strftime('%H%M'))
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	
	with open('%s/cloudmask.png' % out_dir, 'w') as cloudmask_out,\
	     open('%s/radar.png' % out_dir, 'w') as radar_out,\
	     open('%s/cloudmask.wld' % out_dir, 'w') as cloudmask_wld_out,\
	     open('%s/radar.wld' % out_dir, 'w') as radar_wld_out,\
	     open('%s/rainbows.json' % out_dir, 'w') as rainbow_out:
		# radar params
		[radar_dx, _, _, radar_dy, radar_nwx, radar_nwy] = [float(v.strip()) for v in radar_wld.split('\n')]
		radar_nw = np.asarray([radar_nwx, radar_nwy])
		radar_dv = np.asarray([radar_dx, radar_dy])
		
		metar_lookup = [feature['properties'] for feature in metar_json['features']]
		metar_kd = KDTree([feature['geometry']['coordinates'] for feature in metar_json['features']])
		
		# add visilibity? Eh, maybe not, might be cheaper to still do the point-in-polygon thing anyways since it's per-point
		# note: we're also considering inner contours too. Not likely to have big holes, but then it won't be expensive anyways
		_, radar_contours, _ = cv2.findContours(radar_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
					sky_intercept = travel(
						P * math.pi / 180,
						math.pi - (pysolar.GetAzimuth(P[1], P[0], now) * math.pi / 180), # pysolar uses latlon; we use lonlat
						ceil * HUNDRED_FEET_TO_KM / math.tan(pysolar.GetAltitude(P[1], P[0], now) * math.pi / 180) / 6370 # TEMP: earth radius will find a better home later
					)
					cloudmask_coords = np.divide(sky_intercept * 180 / math.pi - cloudmask_nw, cloudmask_dv).astype(np.uint16)
					cloudmask_px = cloudmask[cloudmask_coords[1], cloudmask_coords[0]]
					
					if cloudmask_px == 0 or \
						cloudmask_px == 42:
					   rainbow_out.write('[%s, %s],\n' % tuple(P))
					   sky_points.add((int(round(cloudmask_coords[1])), int(round(cloudmask_coords[0]))))
		
		# aesthetic changes to source images
		radar_src_colorramp = [
			( 0, 0, 0, 0 ),
			( 236, 236, 0, 0 ),
			( 246, 160, 1, 0 ),
			( 246, 0, 0, 0 ),
			( 0, 255, 0, 0 ),
			( 0, 200, 0, 0 ),
			( 0, 144, 0, 0 ),
			( 0, 255, 255, 0 ),
			( 0, 192, 231, 0 ),
			( 0, 144, 255, 0 ),
			( 0, 0, 255, 0 ),
			( 0, 0, 214, 0 ),
			( 0, 0, 192, 0 ),
			( 255, 0, 255, 0 ),
			( 153, 85, 201, 0 )
		]
		radar_grayscale_colorramp = [int(round(0.299 * d[2] + 0.144 * d[0] + 0.587 * d[1])) for d in radar_src_colorramp]
		radar_dest_colorramp = [
			( 0, 0, 0, 0),
			( 203, 234, 249, 255 ),
			( 173, 220, 242, 255 ),
			( 157, 207, 230, 255 ),
			( 139, 200, 228, 255 ),
			( 112, 179, 218, 255 ),
			( 85, 160, 207, 255 ),
			( 62, 148, 202, 255 ),
			( 45, 130, 195, 255 ),
			( 27, 111, 183, 255 ),
			( 10, 83, 157, 255 ),
			( 6, 64, 130, 255 ),
			( 4, 49, 104, 255 ),
			( 4, 32, 104, 255 ),
			( 3, 23, 81, 255 )
		]
		radar_out_img = np.zeros(radar.shape[0:2] + (4,))
		for i, color in enumerate(radar_grayscale_colorramp):
			radar_out_img += np.multiply((radar_gray == color).astype(np.uint8).reshape(radar_gray.shape + (1,)), np.tile(
					radar_dest_colorramp[i],
					radar_gray.shape + (1,)
				))
			# mask = np.ma.array(
			# 	np.tile(
			# 		np.tile(
			# 			radar_dest_colorramp[i],
			# 			radar.shape[1]
			# 		),
			# 		radar.shape[0]
			# 	),
			# 	mask = np.equal(
			# 		np.concatenate(
			# 			(radar, np.zeros(radar.shape[0:2] + (1,))),
			# 			axis=2
			# 		),
			# 		color
			# 	)
			# )
			# radar_out_img += mask
			print(color)
		scipy.misc.imsave(radar_out, radar_out_img)
		
		cloudmask_out_img = np.concatenate((
			np.zeros(cloudmask.shape + (3,)),
			np.reshape(
				np.logical_and(cloudmask != 0, cloudmask != 42).astype(np.uint8) * 42, # alpha = 42/255
				cloudmask.shape + (1,)
			)
		), axis=2)
		scipy.misc.imsave(cloudmask_out, cloudmask_out_img)
		
		cloudmask_wld = '''%f
0
0
%f
%f
%f''' % ((cloudmask_bounds[3] - cloudmask_bounds[2]) / cloudmask.shape[1], (cloudmask_bounds[1] - cloudmask_bounds[0]) / cloudmask.shape[0], cloudmask_bounds[2], cloudmask_bounds[0])
		cloudmask_wld_out.write(cloudmask_wld)
		
		radar_wld_out.write(radar_wld)