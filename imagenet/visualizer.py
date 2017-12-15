import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

import os

def get_df_from_file(file, columns):
	with open(file) as f:
		bbox = f.read()

	bbox = bbox.split('\n')

	bbox = [a.split('\t') for a in bbox ]

	df_boundingbox = pd.DataFrame(bbox, columns=columns)

	return df_boundingbox


root_dir = 'data/tiny-imagenet-100-A/train'

synset_list = os.listdir(root_dir)

df_words = get_df_from_file('data/tiny-imagenet-100-A'+'/words.txt', [ 'synset', 'word_list' ])

for synset in synset_list:

	cur_dir = root_dir +'/'+ synset

	im_list = os.listdir(cur_dir+'/images')

	with open(cur_dir+'/'+synset+"_boxes.txt") as f:
		bbox = f.read()

	df_boundingbox = get_df_from_file(cur_dir+'/'+synset+"_boxes.txt", ['file', 'x', 'y', 'width', 'height'])

	for im in im_list:

		img = Image.open(cur_dir+'/images/'+im)

		bbox = df_boundingbox[df_boundingbox['file'] == im]

		# print img.format, img.size, img.mode
		# print bbox
		# print df_words[df_words['synset'] == synset]

		bbox = [ bbox['x'], bbox['y'], bbox['width'], bbox['height'] ]

		draw = ImageDraw.Draw(img)
		draw.rectangle(bbox)
		del draw

		plt.imshow(img)
		plt.show()

	break

	# bbox = np.array(bbox)

	# bbox = bbox.reshape([bbox.shape[0], bbox[0].shape[0]])