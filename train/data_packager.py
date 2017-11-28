import glob
import re
import tensorflow as tf
from matplotlib.image import imread

OUTPUT_FILE = "output_batch_1"
# OUTPUT_FILE = "data_batch_1.bin"

# use pickle to package image and labels
def package_data(dir):
    # for each image of 32x32 and its corresponding label, create a byte array b[]
    # b's length is 1+32x32x3=3073
    # where b[0] is label, b[1]-b[3072] represents the r/g/b(3 bytes) value of each pixel{32*32)
    labels = open(str(dir) + "/output.csv", "r").readlines()
    images = glob.glob(dir + "/*.jpg")
    images.sort(key=alphanum_key)

    # label_array = [None] * (record_number * (1 + 32 * 32 * 3))

    out_file = open(OUTPUT_FILE, 'wb')

    for i, label in enumerate(labels):
        out_file.write(int(label).to_bytes(1, byteorder='big'))
        image = imread(images[i])
        for color_value in create_color_array(image):
            out_file.write(int(color_value).to_bytes(1, byteorder='big'))

    out_file.close()


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
    package_data('./sampleData')
    # try_batch()

if __name__ == '__main__':
    main()
