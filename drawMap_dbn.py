#import sklearn.mixture as sm
import sklearn.linear_model as sl
import numpy as np
import cPickle as pkl
from PIL import Image, ImageDraw
import scipy
import sys


colors = ([1., 0., 0.],
		 [0., 1., 0.],
		 [0., 0., 1.],
		 [1., 1., 0.],
		 [0., 1., 1.],
		 [1., 0., 1.],
		 [0.25, 0.25, 0.5],
		 [0.5, 0.25, 0.25],
		 [0.25, 0.5, 0.25],
		 [1./3, 1./3, 1./3],)

cmap = [np.array(color) for color in colors]
cell_size = (10, 10)
im_size = (28, 28)
map_size = (25, 40)

for k in range(3):
	classifier = pkl.load(open('output_dbn/dbn_softmax%d.pkl'%k, 'rb'))

	coef = classifier.coef_
	length = (coef ** 2).sum(axis=0)

	c = coef / np.sqrt(length+1e-8)

	colors = c.argmax(axis=0).reshape(map_size)
	d = c.std(axis=0)
	d_r = d.reshape(map_size)
	im = Image.new(mode='RGB', size=(map_size[0]*cell_size[0], map_size[1]*cell_size[1]), color=(255, 255, 255))

	draw = ImageDraw.Draw(im)
	for i in range(map_size[0]):
		for j in range(map_size[1]):
			point = (i*cell_size[0], j*cell_size[1])
			fill = ((1-(cmap[colors[i][j]]*d_r[i][j]))*255)
			fill = fill.astype('int64')
			fill = tuple(fill)
			draw.rectangle((point[0], point[1], point[0]+cell_size[0], point[1]+cell_size[1]), fill = fill)

	im.save(open("output_dbn/map%d.png"%k, 'wb'), "PNG")




def get_lower_rep(h_, W, v):
	return np.dot(W, h_)


all_params = pkl.load(open('output_dbn/pre_trained_dbn_layers.pkl', 'rb'))

W = [l[0] for l in all_params]
h = [l[1] for l in all_params]
v = [l[2] for l in all_params]


for j in range(3):
	cols = []
	rows = []
	for i in range(W[j].shape[1]):
		h_ = np.zeros(W[j].shape[1])
		h_[i] = 1.
		
		for k in reversed(range(j+1)):
			h_ = get_lower_rep(h_, W[k], v[k])
		im = (-h_).reshape(im_size)
		im -= im.min()
		im /= im.max()
		if len(cols) != map_size[1]:
			cols.append(im)
		else:
			rows.append(np.concatenate(cols, axis = 1))
			cols = []
		scipy.misc.toimage(im).resize((im_size[0]*3, im_size[1]*3)).save('output_dbn/layer%d/hf%03d.png'%(j,i))
	im = np.concatenate(rows, axis = 0)
	scipy.misc.toimage(im).save('output_dbn/layer%d/all_l%d.png'%(j, j))
