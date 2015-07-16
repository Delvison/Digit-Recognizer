#import sklearn.mixture as sm
import sklearn.linear_model as sl
import numpy as np
import cPickle as pkl
from PIL import Image, ImageDraw
import scipy

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

classifier = pkl.load(open('output_baseline/baseline.pkl', 'rb'))

coef = classifier.coef_
length = (coef ** 2).sum(axis=0)

c = coef / np.sqrt(length+1e-8)

colors = c.argmax(axis=0).reshape(im_size)
d = c.std(axis=0)
d_r = d.reshape(im_size)


im = Image.new(mode='RGB', size=(im_size[0]*cell_size[0], im_size[1]*cell_size[1]), color=(255, 255, 255))

draw = ImageDraw.Draw(im)
for i in range(im_size[0]):
	for j in range(im_size[1]):
		point = (i*cell_size[0], j*cell_size[1])
		fill = ((1-(cmap[colors[i][j]]*d_r[i][j]))*255)
		fill = fill.astype('int64')
		fill = tuple(fill)
		draw.rectangle((point[0], point[1], point[0]+cell_size[0], point[1]+cell_size[1]), fill = fill)

im.save(open("output_baseline/map.png", 'wb'), "PNG")



for i in range(10):
	scipy.misc.toimage((-coef[i, :]).reshape(im_size)).resize((im_size[0]*3, im_size[1]*3)).save('output_baseline/act_pattern%d.png'%i)

