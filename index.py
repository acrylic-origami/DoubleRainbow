from __future__ import print_function
import itertools
import datetime
import json
import sys
import math

from scipy.spatial import KDTree
import numpy as np
import Pysolar.solar as pysolar
# import requests
import cv2
import gdal
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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
	return (\
		origin[0] - math.asin(math.sin(heading) * math.sin(distance) / math.cos(lat)) % (math.pi * 2),\
		lat\
	)

class GOES16Cloudmask:
	def __init__(self):
		# physical properties
		self.RHO = 6370
		self.ALTITUDE = 35790
		self.ORIGIN = [ -math.pi / 2, 0 ]
		self.ALMOST_PHI = [ float(self.RHO) / (self.ALTITUDE + self.RHO) * math.sqrt(math.pow(self.ALTITUDE + self.RHO, 2) - math.pow(self.RHO, 2)), ((math.pow(self.ALTITUDE + self.RHO, 2) - math.pow(self.RHO, 2)) / (self.ALTITUDE + self.RHO)) ]
		
		# image properties
		self.IMG_MIDPOINT = np.asarray([313, 356])
		self.GLOBE_DIAMETER = 655
		
	def _sat_angle(self, alpha):
		return math.atan2(alpha * self.ALMOST_PHI[0], self.ALMOST_PHI[1])
	
	def _cartesian(self, spherical): # expect tuple of lon-lat
		return np.asarray([
			math.cos(spherical[0]) * math.cos(spherical[1]),
			math.sin(spherical[0]) * math.cos(spherical[1]),
			-math.sin(spherical[1])
		])
	
	def deproject(self, img_coords):
		origin_distance = math.sqrt(np.linalg.norm(img_coords - self.IMG_MIDPOINT))
		sat_angle = self._sat_angle(origin_distance / self.GLOBE_DIAMETER * 2)
		great_angle = -sat_angle + math.asin((self.ALTITUDE + self.RHO) / self.RHO * math.sin(sat_angle))
		chord = math.sin(great_angle) * self.RHO
		azimuth_angle = math.atan2(img_coords[1] - self.IMG_MIDPOINT[1], img_coords[0] - self.IMG_MIDPOINT[0])
		lon = math.asin(math.cos(azimuth_angle) * chord / math.sqrt(math.pow(self.RHO, 2) - math.pow(chord * math.sin(azimuth_angle), 2))) + self.ORIGIN[1]
		lat = -math.asin(math.sin(azimuth_angle) * chord / self.RHO) + self.ORIGIN[1]
		return np.asarray([lon, lat])
		
	def project(self, geo_coords):
		rho_vec = self._cartesian(geo_coords) * self.RHO
		n_hat = np.asarray([ 0, -1, 0 ]) # pointing out of 90\deg W at equator
		sat_to_point_vec = rho_vec - np.asarray([ 0, -self.RHO - self.ALTITUDE, 0 ])
		sat_to_point_hat = sat_to_point_vec / np.linalg.norm(sat_to_point_vec)
		return (((math.pow(self.RHO, 2) / (self.ALTITUDE + self.RHO)) - (np.dot(rho_vec, n_hat)) / np.dot(sat_to_point_hat, n_hat)) * sat_to_point_hat + rho_vec)[0::2] / self.RHO * self.GLOBE_DIAMETER / 2 + self.IMG_MIDPOINT
		
if __name__ == '__main__':
	HUNDRED_FEET_TO_KM = 0.3048 * 100 * 0.001

	# sys.argv expect: [__FILE__, cloudmask, radar tiff, metar json, elevation]
	f_cloudmask, f_radar, f_metar, f_out = sys.argv[1:] # f_elevation
	f_radar_wld = '%s.wld' % '.'.join(f_radar.split('.')[:-1])
	# need to open radar with image lib to convert to greyscale; openCV?
	radar = cv2.imread(f_radar, cv2.IMREAD_GRAYSCALE)
	with open(f_radar_wld, 'r') as radar_wld, open(f_metar, 'r') as metar, open(f_out, 'a') as outfile:
		# radar params
		[dx, _, _, dy, nwx, nwy] = [float(v.strip()) for v in list(radar_wld)]
		nw = np.asarray([nwx, nwy])
		dv = np.asarray([dx, dy])
		
		# GOES-16 cloudmask helpers
		proj = GOES16Cloudmask()
		globe_mask = cv2.imread('data/globe_mask.png', cv2.IMREAD_GRAYSCALE)
		markings = cv2.imread('data/markings.png', cv2.IMREAD_GRAYSCALE)
		border_mask = cv2.imread('data/borders.png', cv2.IMREAD_GRAYSCALE)
		
		cloudmask = cv2.imread(f_cloudmask)
		cardinal_directions = itertools.permutations([-1, 0, 1], 2)
		for i, row in enumerate(border_mask):
			for j, pix in enumerate(row):
				if pix > 0:
					for direction in cardinal_directions:
						if globe_mask[i + direction[0], j + direction[1]] > 0 and not border_mask[i + direction[0], j + direction[1]]:
							cloudmask[i, j] = cloudmask[i + direction[0], j + direction[1]]
		
		cloudmask = np.logical_and(
			globe_mask > 0,
			np.logical_or.reduce((
				cv2.inRange(cloudmask, (0, 0, 255), (0, 0, 255)), # ice cloud
				cv2.inRange(cloudmask, (0, 255, 255), (0, 255, 255)), # ice cloud weak
				cv2.inRange(cloudmask, (255, 255, 0), (255, 255, 0)), # supercooled water cloud
				cv2.inRange(cloudmask, (255, 0, 0), (255, 0, 0)) # water cloud
			))
		)
		
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
				contour = np.multiply(np.squeeze(px_contour).astype(np.float32), dv) + nw
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
						math.pi + (pysolar.GetAzimuth(P[1], P[0], T) * math.pi / 180), # pysolar uses latlon; we use lonlat
						ceil * HUNDRED_FEET_TO_KM / math.tan(pysolar.GetAltitude(P[1], P[0], T) * math.pi / 180) / proj.RHO # TEMP
					)
					cloudmask_coord = proj.project(sky_intercept)
					cloudmask_px = cloudmask[int(round(cloudmask_coord[1])), int(round(cloudmask_coord[0]))]
					
					if not cloudmask_px:
					   outfile.write('[%s, %s],\n' % tuple(P))
					   sky_points.add((int(round(cloudmask_coord[1])), int(round(cloudmask_coord[0]))))
					
					# clear_flag = True
					# for idx, cloud_poly in enumerate(cloud_polies):
					# 	if cloud_poly.contains(Point(sky_intercept)):
					# 		if contour_depth(idx, cloud_hierarchy) % 2 == 0: # it's a cloud by parity
					# 			clear_flag = False
					# 		break
							
					# if clear_flag:
					# 	# rainbow_points.append(P)
					# 	outfile.write('[%s, %s],\n' % tuple(P))
		
		gdal.Close(cloudmask_raw)
		gdal.Close(lat_raw)
		gdal.Close(lon_raw)
	gdal.Close(cloudmask_hdf)