import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry


def show_mask(mask, ax=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


sam_checkpoint = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\segment-anything\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
# Get the path to the folder containing the images
image_folder = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\PIDNet\data\skyground\Img"

# Create a list of all of the images in the folder
images = os.listdir(image_folder)

# Loop through the images and read them
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))

    # Do something with the image
    # For example, you could display it
    cv2.imshow("Image", img)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    input_point = np.array([[30, 30],[img.shape[1]/2, 30],[img.shape[1]-30,30]])
    input_label = np.array([1,1,1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    best_output = np.max(scores)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        if(best_output == score):
            plt.savefig(os.path.join(image_folder,"masked"+image))
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()