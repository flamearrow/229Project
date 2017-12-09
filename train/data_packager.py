import glob
import os
import re
import tensorflow as tf
from matplotlib.image import imread
OUTPUT_FILE = "output_batch_1"

# how many samples are taken on one note
SAMPLE_COUNT = 40
# how many used for training and test
TRAIN_COUNT = 32
TEST_COUNT = 8
NOTE_START = 22
NOTE_END = 108

TRAIN_OUT_FILE = "train_out"
TEST_OUT_FILE = "test_batch.bin"


# package data from dir
def package_data_with_label_file(dir):
    # for each image of 32x32 and its corresponding label, create a byte array b[]
    # b's length is 1+32x32x3=3073
    # where b[0] is label, b[1]-b[3072] represents the r/g/b(3 bytes) value of each pixel{32*32)
    labels = open(str(dir) + "/output.csv", "r").readlines()
    images = glob.glob(dir + "/*.jpg")
    images.sort(key=alphanum_key)


    out_file = open(OUTPUT_FILE, 'wb')

    for i, label in enumerate(labels):
        out_file.write(int(label).to_bytes(1, byteorder='big'))
        image = imread(images[i])
        for color_value in create_color_array(image):
            out_file.write(int(color_value).to_bytes(1, byteorder='big'))

    out_file.close()


# naming: midiIndex_[pmf]_#.jpg,
# e.g
# F_S0_M100_1.jpg
# ...
# F_S0_M100_100.jpg
# F_S0_M48_1.jpg
# ...
# F_S0_M48_100.jpg

def package_data_with_label_in_file_name(sample_data_dir):
    # for each image of 32x32 and its corresponding label, create a byte array b[]
    # b's length is 1+32x32x3=3073
    # where b[0] is label, b[1]-b[3072] represents the r/g/b(3 bytes) value of each pixel{32*32)

    train_out_file = open(TRAIN_OUT_FILE, 'wb')
    test_out_file = open(TEST_OUT_FILE, 'wb')
    # build file names
    for f in "PMF":
        for note in range(NOTE_START, NOTE_END + 1):
            for sample_index in range(1, SAMPLE_COUNT + 1):
                file_name = str(sample_data_dir + "/" + f + "_S0_M" + str(note) + "_" + str(sample_index) + ".jpg")
                if os.path.isfile(file_name):
                    # write to train
                    if sample_index <= TRAIN_COUNT:
                        print("writing " + file_name + " to " + TRAIN_OUT_FILE)
                        train_out_file.write(int(note).to_bytes(1, byteorder='big'))
                        for color_value in create_color_array(imread(file_name)):
                            train_out_file.write(int(color_value).to_bytes(1, byteorder='big'))
                    # write to test
                    else:
                        print("writing " + file_name + " to " + TEST_OUT_FILE)
                        test_out_file.write(int(note).to_bytes(1, byteorder='big'))
                        for color_value in create_color_array(imread(file_name)):
                            test_out_file.write(int(color_value).to_bytes(1, byteorder='big'))
                else:
                    print(file_name + " doesn't exist")
    train_out_file.close()
    test_out_file.close()


def create_color_array(image):
    (height, width, c) = image.shape
    # height = width = 32, c=3
    # first 1024: red, second 1024: green, third 1024: blue
    # color_array = np.arange(32*32*3)
    color_array = list()
    for color in range(c):
        for y in range(height):
            for x in range(width):
                color_array.append(image[y, x, color])
    return color_array


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def try_batch():
    filename_queue = tf.train.string_input_producer([OUTPUT_FILE])

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth

    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # ccen: First byte is label(uint8), following 3 * 32 * 32 (int32) represents color of each pixel

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

def main():
    package_data_with_label_in_file_name('./sampleData')
    # try_batch()

if __name__ == '__main__':
    main()
