from skimage import data, segmentation, color, io
from skimage.future import graph
from matplotlib import pyplot as plt


def show_rag_cuts():
	img = io.imread('/home/avikalp/PycharmProjects/SIGIR/Data/photonet_dataset/images/17.jpg')


	labels1 = segmentation.slic(img, compactness=30, n_segments=400)
	out1 = color.label2rgb(labels1, img, kind='avg')

	g = graph.rag_mean_color(img, labels1, mode='similarity')
	labels2 = graph.cut_normalized(labels1, g)
	out2 = color.label2rgb(labels2, img, kind='avg')

	fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

	ax[0].imshow(out1)
	ax[1].imshow(out2)

	for a in ax:
	    a.axis('off')

	plt.tight_layout()
	plt.show()


def get_rag_cuts(img):
	labels1 = segmentation.slic(img, compactness=30, n_segments=400)
	out1 = color.label2rgb(labels1, img, kind='avg')

	g = graph.rag_mean_color(img, labels1, mode='similarity')
	labels2 = graph.cut_normalized(labels1, g)
	out2 = color.label2rgb(labels2, img, kind='avg')

	return labels1, labels2


if __name__ == "__main__":
	show_rag_cuts()
	return