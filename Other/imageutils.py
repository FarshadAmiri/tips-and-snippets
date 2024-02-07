# -*- coding: utf-8 -*-
"""Utility functions for image processing."""
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import v2 as tv2
import torch
import torchvision
from PIL import Image
import PIL
import cv2
import io


def draw_image_with_boxes(filename, image, boxes):
    """Draws an image with boxes of detected objects."""

    # plot the image
    plt.figure(figsize=(10,10))
    plt.imshow(image)

    # get the context for drawing boxes
    ax = plt.gca()

    # plot each box
    for box in boxes:

        # get coordinates
        x1, y1, x2, y2 = box

        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'red')

        # draw the box
        ax.add_patch(rect)

    # Save the figure
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, dpi = 300, bbox_inches='tight')
    plt.close()


def one_hot_encode(label, label_values):
    """ Converts a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes.
    """

    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    """Transforms a one-hot format to a 2D array with only 1 channel where each
    pixel value is the classified class key.
    """

    x = np.argmax(image, axis = -1)

    return x


def colour_code_segmentation(image, label_values):
    """Given a 1-channel array of class keys assigns colour codes."""

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def save_fig(figname, **images):
    """Saves a list of images to disk."""

    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize = 20)
        plt.imshow(image)
    plt.savefig(f'{figname}.png')
    plt.close()



def resize_img_dir_padding(image_path, height=320, width=320):
    image = Image.open(image_path)
    # image = Image.fromarray(np.uint8(image)).convert('RGB')
    MAX_SIZE = (width, height)
    image.thumbnail(MAX_SIZE)
    image = np.asarray(image)
    y_border = max(height - image.shape[0], 0)
    x_border = max(width - image.shape[1], 0)
    top = y_border // 2
    bottom = y_border - top
    left = x_border // 2
    right = x_border - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))
    return image



def resize_img_padding(image, height, width):
    height, width = map(int, (height, width))
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    MAX_SIZE = (width, height)
    image.thumbnail(MAX_SIZE)
    image = np.asarray(image)
    y_border = max(height - image.shape[0], 0)
    x_border = max(width - image.shape[1], 0)
    top = y_border // 2
    bottom = y_border - top
    left = x_border // 2
    right = x_border - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))
    return image


def resize_img(image, height, width):
    height, width = map(int, (height, width))
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    MAX_SIZE = (width, height)
    image.thumbnail(MAX_SIZE)
    image = np.asarray(image)
    return image


def annotated_image_numpy(image, boxes):
    """Draws an image with boxes of detected objects."""

    # plot the image
    fig = plt.figure(figsize=(10,10))
    fig.imshow(image)

    # get the context for drawing boxes
    ax = fig.gca()

    # plot each box
    for box in boxes:

        # get coordinates
        x1, y1, x2, y2 = box

        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'red')

        # draw the box
        ax.add_patch(rect)

    # Save the figure
    ax.xticks([])
    ax.yticks([])


    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=300)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    # plt.savefig(filename, dpi = 300, bbox_inches='tight')
    plt.close()

    return img_arr



def image_inference_annotation(image_or_dir, bbox_list, lengths=None, scores=None, output_img_name=None):
    if type(image_or_dir) == str:
        image = Image.open(image_or_dir)
    elif type(image_or_dir) == np.ndarray:
        image = Image.fromarray(np.uint8(image_or_dir)).convert('RGB')
    elif type(image_or_dir) != Image.Image:
        raise TypeError("image_or_dir should be whether a np.ndarray or PIL.Image.Image or a directory string")
    else:
        image = image_or_dir

    # plot the image
    # plt.figure(figsize=(10,10))
    plt.imshow(image)

    # get the context for drawing boxes
    ax = plt.gca()

    # plot each box
    for box in bbox_list:

        # get coordinates
        x1, y1, x2, y2 = box

        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'red')

        # draw the box
        ax.add_patch(rect)

    # Save the figure
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_img_name, dpi = 300, bbox_inches='tight')
    plt.close()
    return



# def draw_bboxes(image, bbox_list, score_list, length_list, output_file_name):
#     # Convert the image to BGR format
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Iterate through the bounding boxes and draw them on the image
#     for i, bbox in enumerate(bbox_list):
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

#         # Draw the score and length on the bounding box
#         score = score_list[i]
#         length = length_list[i]
#         text = f"Score: {score}, Length: {length}"
#         cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     # Save the image to the output file
#     cv2.imwrite(output_file_name, image)


def draw_bboxes(image, bbox_list, score_list, output_file_name, length_list=None):
    # Convert the image to BGR format
    w, h= image.size
    thick = int((h + w) // 900)

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    score_list = [float('%.2f' % elem) for elem in score_list]
    # Iterate through the bounding boxes and draw them on the image
    for i, bbox in enumerate(bbox_list):
        x1, y1, x2, y2 = map(int, bbox)

        top_left = (x1+(x2-x1)*0.2, y1+(y2-y1)*0.8)
        bottom_right = (x1+(x2-x1)*0.8, y1+(y2-y1)*0.2)

        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        print(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thick//3)

        # Draw the score annotation on the bounding box
        score = score_list[i]
        
        text = str(score)

        # If length_list is not None, draw the length annotation on the bounding box
        if length_list is not None:
            print("x1, y1, x2, y2: ", x1, y1, x2, y2)
            length = length_list[i]
            text = text + f"  L={length}"
            # cv2.putText(image, f"L={length}", (int((x1+x2)*0.5), int((y1+y2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 240, 255), 2)
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 240, 255), thick//3)

    # Save the image to the output file
    cv2.imwrite(output_file_name, image)

    return image



def draw_bboxes_2(image, bbox_list, score_list, length_list=None, output_file_name=None):
    """
    Draw bounding boxes on an image with their corresponding scores and lengths.

    Parameters:
        image (PIL.Image.Image): The image to draw bounding boxes on.
        bbox_list (list): A list of bounding boxes in the format [[x1, y1, x2, y2], ...].
        scores (list): A list of scores corresponding to each bounding box.
        lengths (list): An optional list of lengths corresponding to each bounding box.
        output_file_name (str): The name of the output file to save the annotated image to.

    Returns:
        PIL.Image.Image: The annotated image.
    """
    # Create a copy of the original image
    image_copy = image.copy()

    # Draw bounding boxes and scores
    for i, bbox in enumerate(bbox_list):
        x1, y1, x2, y2 = bbox
        score = score_list[i]
        label = f"{score:.2f}"


        # Load the image
        image = Image.open("image.png")

        # Set the pixel values directly
        for x in range(image.width):
            for y in range(image.height):
                image.putpixel((x, y), (255, 0, 0))

        # Save the updated image
        image.save("updated_image.png")
                # Draw bounding box

        image_copy.paste(label, (x1, y1, x2, y2))

        # Draw score
        image_copy.text((x1, y1), label, font=PIL.ImageFont.truetype("arial", 16), fill=(255, 0, 0))

        # Draw length
        if length_list is not None:
            length = length_list[i]
            label = f"Length: {length}"
            image_copy.text((x1, y2), label, font=PIL.ImageFont.truetype("arial", 16), fill=(255, 0, 0))

    # Save annotated image
    if output_file_name is not None:
        image_copy.save(output_file_name)

    return image_copy



def draw_bbox_torchvision(image, bboxes, scores, lengths=None, ships_coords=None, annotations=["score", "length", "coord"], save=True,
                          image_save_name=None, output_annotated_image=False, font_size=14, font=r"calibri.ttf", bbox_width=2, constraints=None):
    # w, h = image.size
    # thick = int((h + w) // 512)
    # font_size = int((h + w) // 64)
    # if font_size == 12:
    #     n_space = 7
    # elif font_size == 10:
    #     n_space = 10

    ####### Constraints ########
    if constraints is not None:
        constraint_terms = constraints.keys()
    else:
        constraint_terms = [None]

    if "length" in constraint_terms:
        l_min = constraints["length"][0]
        l_max = constraints["length"][1]
    else:
        l_min = 0
        l_max = 700
    
    constraints_mask = [((length >= l_min) and (length <= l_max)) for length in lengths]

    lengths = [length for idx, length in enumerate(lengths) if constraints_mask[idx] == True ]
    bboxes =  np.array([bbox for idx, bbox in enumerate(bboxes) if constraints_mask[idx] == True ])
    scores =  np.array([score for idx, score in enumerate(scores) if constraints_mask[idx] == True ])
    ships_coords =  [ships_coord for idx, ships_coord in enumerate(ships_coords) if constraints_mask[idx] == True ]
    ####### Constraints ########

    n_space = int(80 / font_size ) + 2

    colors = [(255, 255, 255), (150, 255, 150), (255, 130, 0), (240, 240, 0), (200, 70, 255),  (0, 255, 0), (200, 255, 180), 
               (40, 210, 150), (140, 250, 15), (230, 255, 100), (200, 230, 255), (15, 255, 230), (255, 150, 0), (255, 255, 255),
              (251, 252, 11),  (40, 220, 10),  (220, 220, 0),(40, 210, 150), (230, 255, 100), (15, 255, 230), ]
    while len(colors) < len(scores):
        colors.append(colors[random.randint(0,len(colors)-1)])

    # Convert PIL.Image.Image to a torch.tensor
    array = np.asarray(image)
    image_tensor = array.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.uint8)
    image_tensor = torch.from_numpy(image_tensor)

    bboxes = torch.from_numpy(bboxes)

    # Generating labels
    labels = [" "*n_space for idx in range(len(scores))]
    if "score" in annotations:
        labels = [f"{labels[idx]}\n{' '*n_space}Score: {scores[idx]:.2f}  "  for idx in range(len(labels))]
    if "length" in annotations:
        if lengths != None:
            labels = [f"{labels[idx]}L: {lengths[idx]:.0f}" for idx in range(len(labels))]
    if "coord" in annotations:
        if ships_coords !=None:
            ships_coords = tuple(map(lambda x: (round(x[0], 4), round(x[1], 4)), ships_coords))
            labels = [f"{labels[idx]}\n{' '*n_space}Lat: {ships_coords[idx][1]}" for idx in range(len(labels))]
            labels = [f"{labels[idx]}\n{' '*n_space}Lon: {ships_coords[idx][0]}" for idx in range(len(labels))]

    # draw bounding boxes with fill color
    
    try:
        try:
            annotated_image= draw_bounding_boxes(image_tensor, bboxes, width=bbox_width, labels= labels, font_size=font_size, font=font, colors=colors[:len(scores)])
        except:
            annotated_image= draw_bounding_boxes(image_tensor, bboxes, width=bbox_width, labels= labels, colors=colors[:len(scores)])
    except:
        annotated_image = image_tensor


    annotated_image = torchvision.transforms.ToPILImage()(annotated_image)
    if save:
        if image_save_name == None:
            raise ValueError("image_save_name must be provided when 'save' parameter is set True")
        annotated_image.save(image_save_name)

    if output_annotated_image:
        return annotated_image