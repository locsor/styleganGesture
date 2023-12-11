"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import torchvision
import MiDaS.utils as utils
import cv2
import argparse
import time
import pickle

from datetime import datetime, timedelta, date

from pynput import keyboard

import numpy as np

# from imutils.video import VideoStream
from MiDaS.midas.model_loader import default_models, load_model

# from hand.data import cfg
# from hand.layers.functions.prior_box import PriorBox
# from hand.models.faceboxes import FaceBoxes
# from hand.utils.box_utils import decode

import sys
sys.path
sys.path.append('./yolov7')

from yolov7 import load
from yolov7.utils.plots import plot_one_box
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression
from yolov7.utils.general import apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

import gen_images

from sklearn.cluster import KMeans

first_execution = True

def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def hands_yolo(img, model):
    img = cv2.resize(img, (640, 640))

    img = torch.from_numpy(img.copy()).to(device)
    img = img.unsqueeze(0)
    img = torch.moveaxis(img, -1, 1)
    img = img.float()
    img /= 255.0

    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, False)[0]

    conf_thres = 0.2
    iou_thres = 0.001
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, agnostic=True)

    return pred, img

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_hand_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def cluster(centers):
    # dist = np.linalg.norm(centers - centers[:,None], axis=-1)
    dist = np.array([np.linalg.norm(centers[0]-center) for center in centers])
    # ind = dist.argsort(axis = 1)
    # thresh = (dist[:,-1] - dist[:, 0])/2
    # clusters = np.zeroslike(dist)
    # clusters[dist < thresh] = 1
    print(dist)
    clusters = [0] * len(centers)
    if len(centers) > 1:
        if np.max(dist) > 150:
            kmeans = KMeans(n_clusters=2, n_init=5).fit(centers)
            clusters = list(kmeans.labels_)

        if 1 in clusters:
            ind0 = clusters.index(0)
            ind1 = clusters.index(1)
            clusters = np.array(clusters)
            if centers[ind0, 0] > centers[ind1, 0]:
                clusters = 1 - clusters

    return np.array(clusters)

def centeroid(a):
    length = a.shape[0]
    sum_x = np.sum(a[:, 0])
    sum_y = np.sum(a[:, 1])
    return int(sum_x/length), int(sum_y/length)

def set_cameres(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap

def work(cap, model_type, optimize, z, xy, logDict, frame_ct):
    global first_run
    global pressedFlag
    global lockFlag
    global depth_initial
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = frame[40:-40, 320:-320]

    frameDepth = frame.copy()[...,::-1]
    frameYOLO = frame.copy()[...,::-1]
    frameOrig = frame.copy()

    conf_thres = 0.5
    iou_thres = 0.01
    colors = [[255,0,0], [0,0,255]]

    font = cv2.FONT_HERSHEY_SIMPLEX

    if logDict:
        logDict[str(frame_ct)] = {}

    if ret:
        new_shape_h, new_shape_w, _ = frameDepth.shape
        frameDepth = np.float32(frameDepth) / 255.0

        image = transform({"image": frameDepth})["image"]

        with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), frameDepth.shape[1::-1],
                                 optimize, False)

        depth, mask = utils.return_depth(prediction.astype(np.float32), frameOrig.shape)
        # depth_norm = utils.norm_depth(prediction.astype(np.float32))
        depth_out = depth.copy()

        xy_hidden = []
        if not pressedFlag: 

            dets, frameYOLO_torch = hands_yolo(frameYOLO.copy(), net)

            xy = []
            for i, det in enumerate(dets):
                det[:, :4] = scale_coords(frameYOLO_torch.shape[2:], det[:, :4], frameOrig.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x,y = plot_one_box(xyxy, depth_out, label=None, color=(255,0,0), line_thickness=1)
                    xy += [[x, y]]

                    if i == 0:
                        if logDict:
                            logDict[str(frame_ct)]["Pos"] = [x, y]
            
            xy = np.array(xy)
            for point in xy:
                cv2.circle(depth_out, point, 1, (0,0,255), 2)
                cv2.putText(depth_out, str(point[0]), point, font, 
                       0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(depth_out, str(point[1]), point-25, font, 
                       0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(depth_out, str(prediction[point[1], point[0]]), point-50, font, 
                       0.5, (0,0,255), 1, cv2.LINE_AA)

        if pressedFlag:
            dets, frameYOLO_torch = hands_yolo(frameYOLO.copy(), net)

            if logDict:
                logDict[str(frame_ct)]["Active"] = "1"

            for i, det in enumerate(dets):
                det[:, :4] = scale_coords(frameYOLO_torch.shape[2:], det[:, :4], frameOrig.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x,y = plot_one_box(xyxy, depth_out, label=None, color=(255,0,0), line_thickness=1)
                    xy_hidden += [[x, y]]

                    if i == 0:
                        if logDict:
                            logDict[str(frame_ct)]["Pos"] = [x, y]

            xy_hidden = np.array(xy_hidden)

            if not lockFlag:
                lockFlag = True
                for point in xy_hidden:
                    depth_initial = prediction[point[1], point[0]]

            if len(xy_hidden) > 0:
                first_run = False

                depth_value = np.float16(prediction[xy_hidden[0][1], xy_hidden[0][0]])
                depth_delta = np.float16(depth_value - depth_initial)

                if depth_delta <= 0:
                    depth_input = ((depth_value / depth_initial ) * 2.45) - 2.45

                if depth_delta > 0:
                    depth_input = (( (depth_value - depth_initial) / (np.max(prediction) - depth_initial) ) * 2.45)

                for point in xy_hidden:
                    cv2.putText(depth_out, str(depth_input), point-50, font, 
                           0.5, (0,0,255), 1, cv2.LINE_AA)

                if logDict:
                    logDict[str(frame_ct)]["Depth"] = 2*depth_input

                gen_img, z = gen_images.generate_images(gen_model, [xy[0][1], xy[0][0], 4*depth_input], z, 2, 1, 'random', './', (0, 0), 0)

                frame_ct += 1
                return depth_out, gen_img, z, xy, logDict, frame_ct

            else:
                frame_ct += 1
                return depth_out, np.zeros((512, 512), dtype = np.uint8), z, xy, logDict, frame_ct
            # elif len(xy_hidden) == 0 and not first_run:
            #     gen_img, z_main= gen_images.generate_images(gen_model, None, z, 2, 1, 'random', './', (0, 0), 0, pca)
            #     return depth_out, gen_img, z, xy
            # elif len(xy_hidden) == 0 and first_run:
            #     return depth_out, np.zeros((512, 512), dtype = np.uint8), z, xy
        else:
            if logDict:
                logDict[str(frame_ct)]["Active"] = "0"


            if logDict:
                logDict[str(frame_ct)]["Depth"] = -1000

            if lockFlag:
                lockFlag = False
            if len(xy) > 0:
                first_run = False
                gen_img, z = gen_images.generate_images(gen_model, None, z, 2, 1, 'random', './', (0, 0), 0)
                frame_ct += 1
                return depth_out, gen_img, z, xy, logDict, frame_ct
            elif len(xy) == 0 and not first_run:
                gen_img, z = gen_images.generate_images(gen_model, None, z, 2, 1, 'random', './', (0, 0), 0)
                frame_ct += 1
                return depth_out, gen_img, z, [], logDict, frame_ct
            elif len(xy) == 0 and first_run:
                frame_ct += 1
                return depth_out, np.zeros((512, 512), dtype = np.uint8), z, [], logDict, frame_ct



def on_press(key):
    global pressedFlag
    if keyboard.Key.space:
        pressedFlag = True

def on_release(key):
    global pressedFlag
    if keyboard.Key.space:
        pressedFlag = False

def run(input_path, output_path, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False, log=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """
    print("Initialize")

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    global model, transform, net_w, net_h
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    global net
    net = load()
    net.eval()

    global gen_model
    gen_model = gen_images.load_moad('', './')

    logDict = {}
    #0 is scare
    seed = 1
    z = np.random.RandomState(seed).randn(1, 512)
    if log:
        logDict["Date"] = str(datetime.now())
        logDict["Seed"] = seed
        logDict["Z0"] = z
        logDict["Img Size"] = [640, 640]

    # from sklearn.decomposition import PCA
    # z_pca = np.reshape(z[0], (32, 16))
    # pca = PCA(n_components=8, svd_solver='auto')
    # z_pca = pca.fit_transform(z_pca)#.flatten()

    global pressedFlag
    pressedFlag = False

    global first_run 
    first_run = True

    global lockFlag
    lockFlag = False

    global depth_initial
    depth_initial = 0

    print("Start processing")

    cap = set_cameres(0)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    xy = None
    frame_ct = 0

    if logDict:
        video0 = cv2.VideoWriter('./log/cam.avi', 
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, (640, 640))
        video1 = cv2.VideoWriter('./log/gen.avi', 
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, (256, 256))
    
    while cap.isOpened(): #and capR.isOpened():

        out, gen_img, z, xy, logDict, frame_ct = work(cap, model_type, optimize, z, xy, logDict, frame_ct)

        if logDict:
            logDict[str(frame_ct-1)]["Z"] = z
            video0.write(out)
            video1.write(gen_img)

        # out, xy = work(cap, model_type, optimize, z, xy)

        # out = cv2.resize(out, (625, 625))
        out = draw_grid(out, (22, 22), color=(0,0,255))

        cv2.putText(out, str(pressedFlag), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('0', out)
        cv2.imshow('1', gen_img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            if logDict:
                with open('./log/log.p', 'wb') as fp:
                    pickle.dump(logDict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            video0.release()
            video1.release()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='dpt_large_384',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )
    parser.add_argument('--log',
                        action='store_true',
                        help='start recording session log'
                        )

    args = parser.parse_args()


    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale, args.log)
