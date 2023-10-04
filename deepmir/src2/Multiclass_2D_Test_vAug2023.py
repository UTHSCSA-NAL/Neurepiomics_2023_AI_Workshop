#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

"""
Perform manual testing with command-line arguments.

This script allows you to perform predictions using a machine learning model on a set of input image files. It takes several command-line arguments to customize the prediction process.

Command-line arguments:
  --img [IMG [IMG ...]]
    Paths to image files to be used for prediction. You can provide multiple image paths.
  -m MODEL, --model MODEL
    Path to the machine learning model file that will be used for prediction (required).
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
    Output directory path where prediction results will be saved (default: "./output_prediction").
  -p PRED_SUFFIX, --pred_suffix PRED_SUFFIX
    Suffix for prediction files (default: "Class_Prediction").
  -cl OUTPUTCLASSES, --outputclasses OUTPUTCLASSES
    Number of output classes (default: 2).
  -d SLICE_DIM, --slice_dim SLICE_DIM
    Dimensions for slicing the input images in the format "width height depth" (default: "256 256 256").
  -n NORMALIZE, --normalize NORMALIZE
    Apply data normalization (default: True).
  -s SCALING, --scaling SCALING
    Apply scaling (default: True).
  -id PTID, --ptid PTID
    Patient ID (default: "NA").
  -gt GT, --gt GT
    Ground truth information (default: "NA").
  -mask BRAIN_MASK, --brain_mask BRAIN_MASK
    Brain mask information (default: "NA").

Example usage:
  python your_script.py --img image1.nii.gz image2.nii.gz -m model.pth
  python your_script.py --img image1.nii.gz image2.nii.gz -m model.pth -o ./predictions -cl 3 -d 128 128 128 -n False -s False -id patient123 -gt ground_truth.nii.gz -mask brain_mask.nii.gz
  
"""
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


import random
import os
import sys
import argparse
import traceback
import datetime

import numpy as np
import nibabel as nib
import tensorflow as tf

# Set TensorFlow log level to ERROR to suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from skimage.filters import threshold_otsu
from tensorflow.python.keras import models

from Multiclass_2D_Utils_vFeb2023 import pad_volume, crop_volume
from RescaleImages import rescaleImage
from metrics import mean_iou

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
random.seed(987)



###########################################################
#   Applying the model to test data and saving results    #
###########################################################

def apply_model(model_path, testline, output_file, data_channels = 1, outputclasses = 2, slice_dim = (256, 256, 256), normalize = False, scaling = False, prediction_file_suffix = "_Class_Prediction.nii.gz"):
	
	print(f"testline: \n{testline}")
	
	output_dirname = os.path.dirname(output_file)
	# Load the model that was saved. This should be the best saved model.
	model = models.load_model(model_path, custom_objects = {'mean_iou': mean_iou})

	# Load the volume channesl, and generate output filenames
	channel_niftii = list()
	channel_img = list()
	channel_outputfilenames = list()

	original_img_shapes = list()
	original_additional = list()
	
	print("\n... processing the images ...")
	for ch in range(data_channels):

		channel_niftii.append(nib.load(testline[ch + 3]))  # 0 is the ptid, 1 is the groundtruth , 2 is the brain_mask
		tmpvol, oshp, oadditional = pad_volume(channel_niftii[ch].get_data(), newshape = slice_dim, slice_axis = 2)
		channel_img.append(tmpvol)
		original_img_shapes.append(oshp)
		original_additional.append(oadditional)
		#channel_outputfilenames.append(output_basename + "_Padded_Channel_" + str(ch) + ".nii.gz")

		# Using Jimit's rescaling
		if (scaling == True and normalize == True):
			channel_img[ch] = rescaleImage(channel_img[ch], minInt = 0, maxInt = 255, perc = 99.0, method = "norm")

	gtshp = []
	maskshp = []
	
	if (testline[2] is not "NA") and (os.path.isfile(testline[2]) == True):
		mask_niftii = nib.load(testline[2])
		padded_mask_img, maskshp, mask_additional = pad_volume(mask_niftii.get_data())
	else:
		mask_niftii = nib.load(testline[3])
		image_data = mask_niftii.get_data()
		# Calculate the Otsu threshold for the entire 3D volume.
		otsu_threshold = threshold_otsu(image_data)
		# Apply the threshold to the entire 3D volume.
		binary_volume = image_data > otsu_threshold
		padded_mask_img, maskshp, mask_additional = pad_volume(binary_volume)
		
	if (testline[1] is not "NA") and (os.path.isfile(testline[1]) == True):
		gt_niftii = nib.load(testline[1])
		padded_gt_img, gtshp, gt_additional = pad_volume(mask_niftii.get_data())

	# Filenames of the thresholded and unthresholded images
	# Create empty output unthresholded images
	raw_probability_filename = list()
	raw_probability_img = list()

	# Create an array to store the model outputs
	model_output = np.zeros((channel_img[0].shape[0], channel_img[0].shape[1], channel_img[0].shape[2], outputclasses))

	for w in range(outputclasses):
		raw_probability_filename.append(output_dirname + "/" + os.path.splitext(os.path.splitext(os.path.basename(output_file))[0])[0] + "_RawProbability_Class_" + str(w + 1) + ".nii.gz")
		raw_probability_img.append(np.zeros((channel_img[0].shape), dtype = float))  # Change to np.float16
		
	class_prediction = np.zeros(channel_img[0].shape, dtype = float)

	print("... predicting ...")
	print("		original image shape	: ", channel_niftii[0].shape)
	print("		padded image shape	: ", channel_img[0].shape)
	
	kth_slice = 0
	while kth_slice < channel_img[0].shape[2]:

		# Create the test patch from all the channels
		test_patch = np.zeros((1, slice_dim[0], slice_dim[1], data_channels))
		for ch in range(data_channels):
			test_patch[0, :, :, ch] = channel_img[ch][:, :, kth_slice]
			

		# Create empty output patches.
		unthresholded_patch = np.zeros((slice_dim[0], slice_dim[1], outputclasses), dtype = float)

		if (np.sum(padded_mask_img[:, :, kth_slice]) > 0):  # ignore blank slices

			# Run the test data through the model
			predicted_patch = model.predict(x = test_patch, batch_size = 1)
			model_output[:, :, kth_slice, :] = np.copy(predicted_patch[0, :, :, :])
			class_prediction[:, :, kth_slice] = np.argmax(predicted_patch[0, :, :, :], axis = -1)

			# Process the output
			for segclass in range(outputclasses):
				unthresholded_patch[:, :, segclass] = np.copy(predicted_patch[0, :, :, segclass])

		for w in range(outputclasses):
			raw_probability_img[w][:, :, kth_slice] = unthresholded_patch[:, :, w]

		kth_slice = kth_slice + 1

	print("... Writing output files ...")
	# Save the class predictions
	# First crop the predition to the original shape
	class_prediction = crop_volume(class_prediction, original_img_shape=original_img_shapes[0], additional = original_additional[0])
	class_prediction_niftii = nib.Nifti1Image(class_prediction, channel_niftii[0].affine, channel_niftii[0].header)
	class_prediction_niftii.set_data_dtype(float)
	nib.save(class_prediction_niftii,output_file)
	print(f"	Class_prediction	: {output_file}")


	# Save the thresholded and unthresholded images for each output class
	for w in range(outputclasses):
		# First crop the raw probability images to the original shape
		raw_probability_img[w] = crop_volume(raw_probability_img[w], original_img_shapes[0], additional = original_additional[ch])
		unthresholded_predicted_niftii = nib.Nifti1Image(raw_probability_img[w], channel_niftii[0].affine, channel_niftii[0].header)
		unthresholded_predicted_niftii.set_data_dtype(float)
		nib.save(unthresholded_predicted_niftii, raw_probability_filename[w])
		print(f"	Raw_probability_images	: {raw_probability_filename[w]}")


############################################################
### Program starts from here
############################################################
def main():
	
	parser = argparse.ArgumentParser(description="Perform manual test with command-line arguments.")
	parser.add_argument('--img', type=str, nargs='*', help='Paths to image files')
	parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model file")
	parser.add_argument("-o", "--output_file", type=str, default="./Output_Prediction_" + datetime.datetime.now().strftime("%b_%d_%Y"), help="Output directory path")
	parser.add_argument("-d", "--slice_dim", type=lambda s: tuple(map(int, s.split())), default=(256, 256, 256))
	parser.add_argument("-n", "--normalize", type=bool, default=True, help="Apply data normalization (default: True)")
	parser.add_argument("-s", "--scaling", type=bool, default=True, help="Apply scaling (default: True)")


	# Parsing the arguments
	args = parser.parse_args()

	# Defining variables using the parsed arguments
	model = args.model
	output_file = str(args.output_file)
	imgs = args.img  # args.img is already a list
	data_channels = len(imgs)
	slice_dim = tuple(args.slice_dim)
	normalize = args.normalize
	scaling = args.scaling

	# Print all parsed arguments
	print("\n....... Parsed arguments .......")
	print(f"Model file		: {model}")
	print(f"Output file		: {output_file}")
	print(f"Data channels		: {data_channels}")
	print(f"Slice dimensions	: {slice_dim}")
	print(f"Normalize		: {normalize}")
	print(f"Scaling			: {scaling}")
	
	print("\n.......Image_files......." + "\n" + str(imgs) + "\n")

	
	#output files
	file_name = os.path.join(os.path.dirname(output_file),os.path.splitext(os.path.splitext(os.path.basename(output_file))[0])[0] + "_" + datetime.datetime.now().strftime("%b_%d_%Y") + ".log")	
		
	if (os.path.exists(output_file) == False):
		print("... creating output directories ...\n")
		os.makedirs(os.path.dirname(output_file))
		
		if os.path.exists(file_name):
			# If it exists, open it in append mode
			log_file = open(file_name, "a")
		else:
			# If it doesn't exist, create and open it in write mode
			log_file = open(file_name, "w")
	else:
		log_file = open(file_name, "w")
		print("\n    WARNING: Found existing prediction for " + output_file + "\n")
		log_file.write(f"processing...{datetime.datetime.now()}\n")
		log_file.write("\n    WARNING: Found existing prediction for " + output_file  + "\n")
		log_file.write("\nDone")
		log_file.write("\n############################################################################################\n")
		return -98
		
	
	print("\nBeginning model application to data...")
	print("processing...\n")
	
	if (os.path.exists(model)):
		log_file.write(f"processing...{datetime.datetime.now()}\n")
		log_file.write("Model		: " + model + "\n")
		log_file.write("Output file		: " + str(output_file) + "\n")
		log_file.write(f"\nImage_files	:\n{str(args.img)}\n")
		
			

		
		# check if images exists
		flag = False
		for idx, img in enumerate(imgs):
			if not (os.path.exists(img)):
				print("\n********ERROR: channel does not exist********")
				print(f"{img}\n")
				log_file.write(f"********ERROR: channel does not exist : {os.path.exists(img)},{img}********\n")
				flag = True
				

		if flag == True:
			return -99
		
		try:
			#creating testline
			testline = ["0"] + ["0"] + ["0"] + imgs

			apply_model(model_path = model,
					testline = testline,
					output_file = output_file,
					data_channels = data_channels,
					slice_dim = slice_dim,
					normalize = normalize,
					scaling = scaling)
			
		except:
			print("\n\nException occurred. Model was not able to predict\n")
			print(sys.exc_info()[0])
			log_file.write("\n\nException occurred. Model was not able to predict\n")
			log_file.write(str(sys.exc_info()[0]))
			log_file.write("\nDone")
			log_file.write("\n############################################################################################\n")
			print("\n\n")

			traceback.print_exc()
			
		
	else:
		print("\n********ERROR: Model File does not exist********")
		print("  Model     : " + model)
		log_file.write("\n********ERROR: Model File does not exist********")
		log_file.write(f"  Model: {model}\n")
		log_file.write("\nDone")
		log_file.write("\n############################################################################################\n")
		return -97
		
	log_file.write("\nDone")
	log_file.write("\n############################################################################################\n")
	log_file.close()
		
		
		
if (__name__ == "__main__"):
	main()
			
