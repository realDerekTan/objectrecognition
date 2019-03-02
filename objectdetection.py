import tensorflow as tf
import os
import sys
import tarfile
import urllib
import inception_preprocessing
import matplotlib.pyplot as plt
import numpy as np
import slim
from nets import inception
import imagenet
import tensorflow.contrib.slim as slim


def download_and_uncompress_tarball(tarball_url, dataset_dir):
	filename = tarball_url.split('/')[-1]
	filepath = os.path.join(dataset_dir, filename)

	def _progress(count, block_size, total_size):
		sys.stdout.write('\r>> Downloading %s %.1f%%' % (
			filename, float(count * block_size) / float(total_size) * 100.0))
		sys.stdout.flush()
	filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
	print()
	statinfo = os.stat(filepath)
	print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


# Base url
TF_MODELS_URL = "http://download.tensorflow.org/models/"

# Links to Inception v3 and v4 CNN models
INCEPTION_V3_URL = TF_MODELS_URL + "inception_v3_2016_08_28.tar.gz"
INCEPTION_V4_URL = TF_MODELS_URL + "inception_v4_2016_09_09.tar.gz"

# Directory to save model checkpoints
MODELS_DIR = "models/cnn"
INCEPTION_V3_CKPT_PATH = MODELS_DIR + "/inception_v3.ckpt"
INCEPTION_V4_CKPT_PATH = MODELS_DIR + "/inception_v4.ckpt"

# Make the model directory if it does not exist
if not tf.gfile.Exists(MODELS_DIR):
	tf.gfile.MakeDirs(MODELS_DIR)

# Download the appropriate model if haven't already done so
if not os.path.exists(INCEPTION_V3_CKPT_PATH):
	download_and_uncompress_tarball(INCEPTION_V3_URL, MODELS_DIR)

if not os.path.exists(INCEPTION_V4_CKPT_PATH):
	download_and_uncompress_tarball(INCEPTION_V4_URL, MODELS_DIR)

root_dir = "images/"

if not tf.gfile.Exists(root_dir):
	tf.gfile.MakeDirs(root_dir)


def process_image(image):
	filename = root_dir + image
	with open(filename, "rb") as f:
		image_str = f.read()

	if image.endswith('jpg'):
		raw_image = tf.image.decode_jpeg(image_str, channels=3)
	elif image.endswith('png'):
		raw_image = tf.image.decode_png(image_str, channels=3)
	else:
		print("Image must be either .jpg or .png")
		return

	image_size = 299  # ImageNet image size, different models may be sized differently
	processed_image = inception_preprocessing.preprocess_image(raw_image, image_size, image_size, is_training=False)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		raw_image, processed_image = sess.run([raw_image, processed_image])

	return raw_image, processed_image.reshape(-1, 299, 299, 3)


def plot_color_image(image):
	plt.figure(figsize=(10, 10))
	plt.imshow(image.astype(np.uint8), interpolation='nearest')
	plt.axis('off')


"""
# using sample images as tests
raw_shark, processed_shark = process_image('White_shark.jpg')
plot_color_image(raw_shark)
print(raw_shark.shape, processed_shark.shape)

raw_eagle, processed_eagle = process_image('Bald-Eagle.jpg')
plot_color_image(raw_eagle)
print(raw_eagle.shape, processed_eagle.shape)
"""


# Plots the raw image and displays the top-10 class predictions
def predict(image, version='V3'):
	tf.reset_default_graph()

	# Process the image
	raw_image, processed_image = process_image(image)
	class_names = imagenet.create_readable_names_for_imagenet_labels()

	# Create a placeholder for the images
	x = tf.placeholder(tf.float32, [None, 299, 299, 3], name="X")

	if version.upper() == 'V3':
		model_ckpt_path = INCEPTION_V3_CKPT_PATH
		with slim.arg_scope(inception.inception_v3_arg_scope()):
			# Set the number of classes and is_training parameter
			logits, end_points = inception.inception_v3(x, num_classes=1001, is_training=False)

	predictions = end_points.get('Predictions', 'No key named predictions')
	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess, model_ckpt_path)
		prediction_values = predictions.eval({x: processed_image})

	try:
		# Add an index to predictions and then sort by probability
		prediction_values = [(i, prediction) for i, prediction in enumerate(prediction_values[0, :])]
		prediction_values = sorted(prediction_values, key=lambda x: x[1], reverse=True)

		# Plot the image
		plot_color_image(raw_image)
		plt.show()
		print("Using Inception_{} CNN\nPrediction: Probability\n".format(version))
		# Display the image and predictions
		for i in range(10):
			predicted_class = class_names[prediction_values[i][0]]
			probability = prediction_values[i][1]
			print("{}: {:.2f}%".format(predicted_class, probability * 100))

	# If the predictions do not come out right
	except:
		print(predictions)


# replace image.jpg with the file name of the image you want the model to predict
image = 'placeholder.jpg'
predict(image, version='V3')
