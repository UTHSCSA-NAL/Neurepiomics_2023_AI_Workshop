import numpy as np

def pad_volume(vol, newshape = (256, 256, 256), slice_axis = 2):
    '''
    This function pads the input volume according to the provided newshape. This function will not pad the axis
    along which slicing will take place.
    For example, if the slicing axis is 2, then this function will pad only axis 0 and 1.
    This function handles both even and odd numbered image dimensions. After the first padding,
    the axis with the odd numbered dimension is padded by 1 on one side only so that the
    final padded image had even numbered dimensions. The return variable "additional" tells which axis had the
    additional padding.

    :param vol: volume whose shape is to be checked/padded
    :param newshape: (int tuple) The new shape of the volume
    :return: a volume whose shape conforms to a specified size
    '''

    volshape = vol.shape
    original_shape = volshape

    additional = [0, 0, 0]

    # Pad the volume using numpy.pad() with padding of whatever the edge value is
    if (volshape[0] < newshape[0]):
        padding = (int)(np.floor((newshape[0] - volshape[0]) / 2))
        if (padding > 0):
            vol = np.pad(vol, ((padding, padding), (0, 0), (0, 0)), mode = "constant", constant_values=((0, 0), (0, 0), (0, 0))) #mode = "edge")

            if (vol.shape[0] % 2 != 0):
                vol = np.pad(vol, ((0, 1), (0, 0), (0, 0)), mode = "constant", constant_values=((0, 0), (0, 0), (0, 0))) #mode = "edge")
                additional[0] = -1
        # pad_size[0] = padding

    if (volshape[1] < newshape[1]):
        padding = (int)(np.floor((newshape[1] - volshape[1]) / 2))
        if (padding > 0):
            vol = np.pad(vol, ((0, 0), (padding, padding), (0, 0)), mode = "constant", constant_values=((0, 0), (0, 0), (0, 0))) #mode = "edge")

            if (vol.shape[1] % 2 != 0):
                vol = np.pad(vol, ((0, 0), (0, 1), (0, 0)), mode = "constant", constant_values=((4330, 0), (0, 0), (0, 0))) #mode = "edge")
                additional[1] = -1
        # pad_size[1] = padding

    if (volshape[2] < newshape[2]):
        padding = (int)(np.floor((newshape[2] - volshape[2]) / 2))
        if (padding > 0):
            vol = np.pad(vol, ((0, 0), (0, 0), (padding, padding)), mode = "constant", constant_values=((0, 0), (0, 0), (0, 0))) #mode = "edge")

            if (vol.shape[2] % 2 != 0):
                vol = np.pad(vol, ((0, 0), (0, 0), (0, 1)), mode = "constant", constant_values=((0, 0), (0, 0), (0, 0))) #mode = "edge")
                additional[2] = -1
        # pad_size[2] = padding


    return vol, original_shape, additional





def crop_volume(padded_img, original_img_shape, additional):
    x_amount = int((padded_img.shape[0] - original_img_shape[0]) / 2)
    y_amount = int((padded_img.shape[1] - original_img_shape[1]) / 2)
    z_amount = int((padded_img.shape[2] - original_img_shape[2]) / 2)

    return padded_img[x_amount:padded_img.shape[0] - x_amount + additional[0], y_amount:padded_img.shape[1] - y_amount + additional[1], z_amount:padded_img.shape[2] - z_amount + additional[2]]




def return_as_list_of_strings(string_data):
    # First remove the square brackets
    s = string_data.replace("[", "").replace("]", "")

    # Then tokenize/split based on the comma
    s = s.split(",")

    # Then create an empty list and the tokens to the list
    ret = list()
    for i in range(len(s)):
        ret.append(s[i])

    return ret



def return_as_list_of_ints(string_data):
    # First remove the square brackets
    s = string_data.replace("[", "").replace("]", "")

    # Then tokenize/split based on the comma
    s = s.split(",")

    # Then create an empty list and the tokens to the list
    ret = list()
    for i in range(len(s)):
        ret.append(int(s[i]))

    return ret




def return_as_boolean(string_data):
    if (string_data.lower() == "true"):
        return True
    else:
        return False


def visual_check_datagen_reader(reader_dataset):
    from matplotlib import pyplot as plt

    for item in range(reader_dataset.__len__()):
        x, y = reader_dataset.__getitem__(item)
        print("Batch:", item)

        for i in range(x.shape[0]):
            plt.figure(figsize = (16, 8))

            plt.subplot(1, 3, 1)
            plt.imshow(x[i, :, :, 0], cmap = "gray")
            plt.title("Channel 0")

            # plt.subplot(2, 3, 2)
            # plt.imshow(x[i, :, :, 1], cmap = "gray")
            # plt.title("Channel 1")
            #
            # plt.subplot(2, 3, 3)
            # plt.imshow(x[i, :, :, 2], cmap = "gray")
            # plt.title("Channel 2")
            #
            # plt.subplot(2, 3, 4)
            # plt.imshow(x[i, :, :, 3], cmap="gray")
            # plt.title("Channel 3")

            plt.subplot(1, 3, 2)
            plt.imshow(y[i, :, :, 0], cmap="gray")
            plt.title("Groundtruth 0")

            plt.subplot(1, 3, 3)
            plt.imshow(y[i, :, :, 1], cmap="gray")
            plt.title("Groundtruth 1")

            plt.show()
            plt.close()
            # #

