'''
Efficientdet demo
'''
import argparse
import cv2
import numpy as np
import os
import time
import copy
from PIL import Image
import PIL.ImageColor as ImageColor
import requests
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from tqdm import tqdm

from effdet import create_model

class Dummy:
    pass





def get_args_parser():
    parser = argparse.ArgumentParser(
        'Test detr on one image')
    parser.add_argument(
        '--img', metavar='IMG',
        help='path to image, could be url',
        default='https://www.fyidenmark.com/images/denmark-litter.jpg')
    parser.add_argument(
        '--save', metavar='OUTPUT',
        help='path to save image with predictions (if None show image)',
        default=None)
    parser.add_argument('--classes', nargs='+', default=['Litter'])
    parser.add_argument(
        '--checkpoint', type=str,
        help='path to checkpoint')
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='device to evaluate model (default: cpu)')
    parser.add_argument(
        '--prob_threshold', type=float, default=0.3,
        help='probability threshold to show results (default: 0.5)')
    parser.add_argument(
        '--video', action='store_true', default=False,
        help="If true, we treat impute as video (default: False)")
    parser.set_defaults(redundant_bias=None)
    return parser



def getArgsInFunc():
    args = Dummy()
    args.checkpoint = '20210130-231654-tf_efficientdet_d2.pth.tar'
    args.img = 'IMG-1703.jpg'
    args.save = './test'
    args.classes = ['Litter']
    args.device='cuda:0'
    args.prob_threshold=0.3
    args.video=False
    return args
    



# standard PyTorch mean-std input image normalization
def get_transforms(im, size=768):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(im).unsqueeze(0)


def rescale_bboxes(out_bbox, size, resize):
    img_w, img_h = size
    out_w, out_h = resize
    b = out_bbox * torch.tensor([img_w/out_w, img_h/out_h,
                                 img_w/out_w, img_h/out_h],
                                dtype=torch.float32).to(
                                    out_bbox.device)
    return b


# from https://deepdrive.pl/
def get_output(img, prob, boxes, classes=['Litter'], stat_text=None):
    # colors for visualization
    STANDARD_COLORS = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige',
        'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
        'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk',
        'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey',
        'DarkKhaki', 'DarkOrange', 'DarkOrchid', 'DarkSalmon',
        'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold',
        'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink',
        'IndianRed', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush',
        'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray',
        'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen',
        'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue',
        'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta',
        'MediumAquaMarine', 'MediumOrchid', 'MediumPurple',
        'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream',
        'MistyRose', 'Moccasin', 'NavajoWhite', 'OldLace', 'Olive',
        'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod',
        'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip',
        'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue',
        'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet',
        'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    palette = [ImageColor.getrgb(_) for _ in STANDARD_COLORS]
    for p, (x0, y0, x1, y1) in zip(prob, boxes.tolist()):
        cl = int(p[1] - 1)
        color = palette[cl]
        start_p, end_p = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(img, start_p, end_p, color, 2)
        text = "%s %.1f%%" % (classes[cl], p[0]*100)
        cv2.putText(img, text, start_p, cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 10)
        cv2.putText(img, text, start_p, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    if stat_text is not None:
        cv2.putText(img, stat_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 10)
        cv2.putText(img, stat_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 3)
    return img



def plot_results(pil_img, prob, boxes, classes=['Litter'],save_path=None, colors=None):



    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """


    def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h+font_scale*3), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    imageData = np.array(pil_img)
    imageOriginal = copy.deepcopy(imageData)



    if colors is None:
        colors = 100 * [
           [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
        left = int(xmin) 
        top = int(ymin) 
        right = int(xmax) 
        bottom = int(ymax) 
        c = np.array(c) * 255
        cl = int(p[1]-1)
        label = f'{classes[cl]}: {p[0]:0.2f}'
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(imageData,(left, top), (right, bottom), c, thick)

        draw_text(imageData, label,
          font=0, #cv2.FONT_HERSHEY_PLAIN,
          pos=(left-10, top - 12),
          font_scale=int(1e-3 * imgHeight),
          font_thickness=thick,
          text_color=(0,0,0),
          text_color_bg=c
          )
    imageData = cv2.cvtColor(imageData, cv2.COLOR_RGB2BGR)
    imageOriginal = cv2.cvtColor(imageOriginal, cv2.COLOR_RGB2BGR)
    if  save_path != None:   
        imageOutputPath = save_path+'.png'
        cv2.imwrite(imageOutputPath, imageData)
        cv2.imwrite(imageOutputPath[:-4]+'org.png', imageOriginal)

    return imageOriginal, imageData, prob, boxes






def set_model(model_type, num_classes, checkpoint_path, device):

    # create model
    model = create_model(
        model_type,
        bench_task='predict',
        num_classes=num_classes,
        pretrained=False,
        redundant_bias=True,
        checkpoint_path=checkpoint_path
    )

    param_count = sum([m.numel() for m in model.parameters()])
    print('Model %s created, param count: %d' % (model_type, param_count))
    model = model.to(device)
    return model


def main(im):
    args = getArgsInFunc()
    # prepare model for evaluation
    torch.set_grad_enabled(False)
    num_classes = len(args.classes)
    model_name = 'tf_efficientdet_d2'
    model = set_model(model_name, num_classes, args.checkpoint, args.device)

    model.eval()
    # get image

    # mean-std normalize the input image (batch-size: 1)
    img = get_transforms(im)

    # propagate through the model
    outputs = model(img.to(args.device))

    # keep only predictions above set confidence
    bboxes_keep = outputs[0, outputs[0, :, 4] > args.prob_threshold]
    probas = bboxes_keep[:, 4:]

    # convert boxes to image scales
    bboxes_scaled = rescale_bboxes(bboxes_keep[:, :4], im.size,
                                   tuple(img.size()[2:]))

    # plot and save demo image
    imageOriginal, imageData, prob, boxes = plot_results(im, probas, bboxes_scaled.tolist(), args.classes, args.save)
    #imageOriginal: original img in BGR np array
    #imageData: data with bounding box in BGR np array
    #prob: probability in list(float)
    #boxes: boxes coord in list([xmin, ymin, xmax, ymax])
    return imageOriginal, imageData, prob, boxes


if __name__ == '__main__':
    args = getArgsInFunc()
    if args.img.startswith('https'):
        im = Image.open(requests.get(args.img, stream=True).raw).convert('RGB')
    else:
        im = Image.open(args.img).convert('RGB')

    if args.video:
        save_frames(args)
    else:
        main(im)
