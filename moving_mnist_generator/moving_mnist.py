import math
import os
import sys
import pickle

import csv
import numpy as np
from PIL import Image


###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################

# helper functions
def arr_from_img(im, mean=0, std=1):
    '''
    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract
    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    '''
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def get_image_from_array(X, index, mean=0, std=1):
    '''
    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[index] + mean) * 255.) * std).reshape(ch, w,
                                                     h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    return load_mnist_images('train-images-idx3-ubyte.gz'), load_mnist_labels('train-labels-idx1-ubyte.gz')


def get_image_from_array_with_label(X, Y, index, label, mean=0, std=1):
    '''
    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    true_index = index
    if Y[index] != label:
        for i in range(index, Y.shape[0]):
            if Y[i] == label:
                true_index = i
                break

    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[true_index] + mean) * 255.) * std).reshape(ch, w,
                                                          h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


def generate_moving_mnist(digits, motions, shape=(64, 64), num_frames=30, num_sequences=1, original_size=28):
    '''
    Args:
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_sequences: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)
    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_images x 1 x new_width x new_height
    '''

    mnist_imgs, mnist_labels = load_dataset()
    width, height = shape

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset of shape of num_frames * num_images x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_sequences, 1,
                       width, height), dtype=np.uint8)
    action_vectors = np.zeros((num_sequences, num_frames, 4), dtype=np.float)

    for img_idx in range(num_sequences):
        # Randomly generate direction, speed and velocity for both images
        # motions = list(np.random.randint(low=0, high=len(digits), size=len(digits)))

        motion_dict = {}
        # circular_motion = {}
        positions = [[] for _ in digits]
        for i in range(len(digits)):
            if motions[i] in ["vertical", "horizontal", "zigzag"]:
                speed = np.random.randint(5) + 2
                theta = 0
                if motions[i] == "vertical":
                    theta = np.pi/2
                elif motions[i] == "zigzag":
                    theta = np.pi*np.random.randn()/2
                motion_dict[i] = {
                    "veloc": [speed*math.cos(theta), speed*math.sin(theta)]
                }
                positions[i].append(np.random.rand() * x_lim)
                positions[i].append(np.random.rand() * y_lim)
            elif motions[i] in ["circular_clockwise", "circular_anticlockwise"]:
                r = np.random.randint(0, x_lim//2)
                theta = np.pi * np.random.randn()/2
                angular_velocity = np.random.rand()
                if motions[i] == "circular_clockwise":
                    angular_velocity *= -1
                motion_dict[i] = {
                    "r": r,
                    "theta": theta,
                    "angular_velocity": angular_velocity
                }
                positions[i].append(
                    width//2 + r*math.cos(theta) - original_size//2)
                positions[i].append(
                    height//2 - r*math.sin(theta) - original_size//2)
            elif motions[i] == "tofro":
                motion_dict[i] = {
                    "center_x": width//2,  # + np.random.randn()*x_lim/4,
                    "center_y": height/2,  # + np.random.randn()*y_lim/4,
                    "size": original_size,
                    "waxing": True,
                    "size_step": np.random.randint(5)
                }
                positions[i].append(
                    motion_dict[i]["center_x"] - motion_dict[i]["size"]//2)
                positions[i].append(
                    motion_dict[i]["center_y"] - motion_dict[i]["size"]//2)

        images = []
        for digit in digits:
            rand_num = np.random.randint(0, int(0.9*mnist_imgs.shape[0]))
            images.append(Image.fromarray(get_image_from_array_with_label(
                mnist_imgs, mnist_labels, rand_num, digit)))

        # Generate new frames for the entire num_frames
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height))
                        for _ in range(len(digits))]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # In canv (i.e Image object) place the image at the respective positions
            # Super impose both images on the canvas (i.e empty np array)
            for i, canv in enumerate(canvases):
                if motions[i] == "tofro":
                    canv.paste(images[i].resize((motion_dict[i]["size"], motion_dict[i]["size"])),
                               (int(positions[i][0]), int(positions[i][1])))
                elif motions[i] == "vertical":
                    canv.paste(
                        images[i], (int(positions[i][0]), int(positions[i][1])))
                    if motion_dict[i]["veloc"][1] >= 0:
                        action_vectors[img_idx][frame_idx][1] = 1
                    elif motion_dict[i]["veloc"][1] < 0:
                        action_vectors[img_idx][frame_idx][0] = 1
                elif motions[i] == "horizontal":
                    canv.paste(
                        images[i], (int(positions[i][0]), int(positions[i][1])))
                    if motion_dict[i]["veloc"][0] >= 0:
                        action_vectors[img_idx][frame_idx][3] = 1
                    elif motion_dict[i]["veloc"][0] < 0:
                        action_vectors[img_idx][frame_idx][2] = 1
                elif motions[i] in ["circular_clockwise", "circular_anticlockwise", "zigzag"]:
                    canv.paste(
                        images[i], (int(positions[i][0]), int(positions[i][1])))
                canvas += arr_from_img(canv, mean=0)

            for i in range(len(digits)):
                if motions[i] in ["vertical", "horizontal", "zigzag"]:
                    for j in range(2):
                        new_pos = positions[i][j] + motion_dict[i]["veloc"][j]
                        if new_pos < -2 or new_pos > lims[j] + 2:
                            motion_dict[i]["veloc"][j] *= -1
                        positions[i][j] += motion_dict[i]["veloc"][j]
                elif motions[i] in ["circular_clockwise", "circular_anticlockwise"]:
                    motion_dict[i]["theta"] += motion_dict[i]["angular_velocity"]
                    r = motion_dict[i]["r"]
                    theta = motion_dict[i]["theta"]
                    positions[i][0] = width//2 + r * \
                        math.cos(theta) - original_size//2
                    positions[i][1] = height//2 - r * \
                        math.sin(theta) - original_size//2
                elif motions[i] == "tofro":
                    if motion_dict[i]["waxing"]:
                        motion_dict[i]["size"] += motion_dict[i]["size_step"]
                        newX = motion_dict[i]["center_x"] - \
                            motion_dict[i]["size"]//2
                        newY = motion_dict[i]["center_y"] - \
                            motion_dict[i]["size"]//2
                        if newX < -2 or newX > (width - motion_dict[i]["size"]) + 2 or newY < -2 or newY > (height - motion_dict[i]["size"]) + 2:
                            motion_dict[i]["waxing"] = False
                    else:
                        motion_dict[i]["size"] -= motion_dict[i]["size_step"]
                        if motion_dict[i]["size"] == original_size:
                            motion_dict[i]["waxing"] = True

            # Add the canvas to the dataset array
            dataset[img_idx * num_frames +
                    frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)

    return dataset, action_vectors


def tack_on(digit, motion, caption):
    caption += '{},'.format(digit)
    if motion == "vertical":
        caption += '0;'
    elif motion == "horizontal":
        caption += '1;'
    elif motion == "circular_clockwise":
        caption += '2;'
    elif motion == "circular_anticlockwise":
        caption += '3;'
    elif motion == "zigzag":
        caption += '4;'
    elif motion == "tofro":
        caption += '5;'
    return caption


def main(digits, motions, dest, csvname, frame_size=64, num_frames=30, num_sequences=1, original_size=28):

    assert len(digits) > 0, "Need at least one digit"

    dat, action_vectors = generate_moving_mnist(shape=(frame_size, frame_size), num_frames=num_frames, num_sequences=num_sequences,
                                                digits=digits, motions=motions, original_size=original_size)

    row = []
    row.append(digits[0])
    row.append(motions[0])
    # caption = tack_on(digits[0], motions[0], '')
    if len(digits) > 1:
        for i in range(1, len(digits)):
            row.append(digits[i])
            row.append(motions[i])
            # caption = tack_on(digits[i], motions[i], caption)

    fcount = len(os.listdir(dest))

    # f = open(os.path.join(dest, 'captions.csv'), 'a')
    with open(os.path.join(dest, csvname), 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(num_sequences):
            image_dir = os.path.join(dest, '{}'.format(fcount))
            fcount += 1
            os.makedirs(image_dir)
            for j in range(num_frames):
                Image.fromarray(get_image_from_array(
                    dat, i*num_frames+j, mean=0)).save(os.path.join(image_dir, '{}.jpg'.format(j)))
            # f.write('{},{}\n'.format(image_dir, caption))
            writer.writerow([image_dir] + row)
            with open(os.path.join(image_dir, 'actions.pkl'.format(i)), 'wb') as action_f:
                pickle.dump(action_vectors[i], action_f)
    file.close()
    # f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    # The 'dest' argument is the directory in which to store the generated GIFs
    # The 'num_gifs' argument is the no. of GIFs to create
    parser.add_argument('--dest', type=str, dest='dest', default='movingmnist')
    # number of sequences to generate
    parser.add_argument('--num_gifs', type=int, dest='num_gifs', default=1)
    args = vars(parser.parse_args(sys.argv[1:]))

    dest = args['dest']
    csvname = args['csvname']
    num_sequences = args['num_gifs']

    # Create directory and the captions file
    if not os.path.exists(dest):
        os.makedirs(dest)

    if not os.path.exists(os.path.join(dest, csvname)):
        open(os.path.join(dest, csvname), 'x')

    allowed_motions = ["vertical", "horizontal", "circular_clockwise",
                       "circular_anticlockwise", "zigzag", "tofro"]

    digits = [0]
    motions = ["vertical", "horizontal"]

    main(digits, motions, dest, num_sequences=num_sequences)
