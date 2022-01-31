from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from DAMSM import RNN_ENCODER

import os
import sys
import time
import random
from random import randrange
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetD
import torchvision.utils as vutils

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)
import re
import time

from flask import Flask, request, Response, send_file, abort, flash

from io import BytesIO

import platform

# Create two constant. They direct to the app root folder and logo upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'dream_interpolations')

# Configure Flask app and the logo upload folder
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Video codecs. Please adjust this to the operating system. See here: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html under 'Saving a video'.
VIDEO_CODEC = "MJPG" # MacOS codec
# VIDEO_CODEC = "DIVX" # Windows and Fedora (Linux) codec
VIDEO_CONTAINER = ".mp4" # MacOS
# VIDEO_CONTAINER = ".mp4" # Other


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/coco.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--port', type=int, help='port', default=5000)
    parser.add_argument('--ssl_cert', type=str, default='')
    parser.add_argument('--ssl_key', type=str, default='')
    args = parser.parse_args()
    return args

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

def save_pil_imgs_to_disk(pil_imgs_list):
    import uuid
    cnt = 0
    folder_name = str(uuid.uuid4())
    mkdir_p(os.path.join(app.config['UPLOAD_FOLDER'], folder_name))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], folder_name, '')

    video = cv2.VideoWriter(full_filename+"movie"+VIDEO_CONTAINER, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (256,256))
    print(full_filename+"movie"+'.mov')
    for pil_img in pil_imgs_list:
        cnt += 1
        pil_img.save(full_filename+str(cnt)+'.jpg', 'JPEG', quality=70)
        video.write(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    video.release()
    cv2.destroyAllWindows()
    return 'success'

def serve_images_as_zip(pil_imgs_list):
    # Doesnt work
    import io
    import zipfile
    from base64 import encodebytes
    zip_byte_array = BytesIO()
    encoded_imgs = []
    zipf = zipfile.ZipFile(zip_byte_array,'w', zipfile.ZIP_DEFLATED)
    for pil_img in pil_imgs_list:
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG')
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
        encoded_imgs.append(encoded_img)
    for img in encoded_imgs:
        zipf.write(img) 
    zipf.close()
    zip_byte_array.seek(0)
    return send_file(zip_byte_array,
            mimetype = 'zip',
            attachment_filename= 'Name.zip',
            as_attachment = True)


# def serve_pil_to_video(pil_imgs_list):
#     videodims = (256,256)
#     print('here')
#     fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
#     print('2')
#     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('download_file', name=filename))

#     video = cv2.VideoWriter(os.path.join(app.config['UPLOAD_FOLDER'],fourcc,30,videodims)
#     print('3')
#     #draw stuff that goes on every frame here
#     for i in range(0, len(pil_imgs_list)):
#         img_io = BytesIO()
#         pil_imgs_list[i].save(img_io, 'JPEG', quality=70)
#         img_io.seek(0)
#         print('4')
#         print(pil_imgs_list[i])
#         # draw frame specific stuff here.
#         video.write(np.array(pil_imgs_list[i]))
#         print('5')
#     video.release()
#     return send_file(res, mimetype='video/mp4')

def generate_embeddings(dream, hidden, batch_size):
    input_dream = dream
    input_dream = re.sub(r'[^A-Za-z0-9 ]+', '', input_dream)
    input_dream = input_dream.lower().split()
    input_dream = [word for word in input_dream if word in dataloader.dataset.wordtoix]
    print("Dream length (after filtering):", len(input_dream))
    print(input_dream)
    vector_dream = [dataloader.dataset.wordtoix[w] for w in input_dream]
    captions = torch.empty(batch_size, len(vector_dream), dtype=torch.long)
    cap_lens = torch.empty(batch_size, dtype=torch.long)
    if torch.cuda.is_available():
        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
    captions[0] = torch.Tensor(vector_dream)
    cap_lens = cap_lens.tolist()
    cap_lens[0] = len(input_dream)
    cap_lens = torch.Tensor(cap_lens)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return words_embs, sent_emb


def sampling(dream, text_encoder, netG, dataloader,device):

    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    netG.load_state_dict(torch.load('models/%s/netG.pth'%(cfg.CONFIG_NAME),map_location=device))
    netG.eval()

    batch_size = 1 #cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)
    cnt = 0
    time_sum = 0

    print("Dream: ", dream)

    

    hidden = text_encoder.init_hidden(batch_size)
    # words_embs: batch_size x nef x seq_len
    # sent_emb: batch_size x nef
    delimiter = "§§"
    if delimiter in dream: # Interpolation between dreams separated by "§§"
        print("Found'§§' delimiter. Splitting into dream sequence...")
        sent_embs_list = []
        interp_embs_list = []
        im_list = []
        N = 90 # Amount of images generated. Divide by 30 to get the length of the interpolation time between two images in seconds.
        dreams_list = dream.split(delimiter)
        # Remove empty strings from dreams
        dreams_list = [i for i in dreams_list if i]
        print('dreams_list: ', dreams_list)
        for i in range(len(dreams_list)):
            words_embs, sent_emb = generate_embeddings(dreams_list[i], hidden, batch_size)
            sent_embs_list.append(sent_emb)
        for i in range(len(sent_embs_list)-1):
            w = 0
            for j in range(N + 1): # Interpolate tensors
                inter_tensor = torch.lerp(sent_embs_list[i], sent_embs_list[i+1], w)
                print('w: ', w)
                print("j: ", j)
                # Interpolate based on sigmoid function
                w = sigmoid_curve(N/2, N, j)
                # w = w + (randrange(80, 120)/10000) # Noise
                # w += 1/N
                interp_embs_list.append(inter_tensor)
        for emb in interp_embs_list:
            im_list.append(generate_image(torch.tensor(emb), save_dir))
        return im_list

def sigmoid_curve(shift, range, x):
    # See https://www.desmos.com/calculator/jl5ybycfqf to understand parameters
    range_base = 20 # See link above for this arbitrary parameter
    return 1/(1+(1+range_base/range)**-(x-shift))


def generate_image(sent_emb, save_dir):
    #######################################################
    # (2) Generate fake images
    ######################################################
    cnt = 0
    time_sum = 0
    start_time = time.time()

    with torch.no_grad():
        noise = torch.randn(batch_size, 100) * 0 # Disable noise by multiplying it with 0
        sent_emb = sent_emb
        noise=noise.to(device)
        fake_imgs = netG(noise,sent_emb)
    for j in range(batch_size):
        im = fake_imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)

        # time it
        elapsed_time = time.time() - start_time
        time_sum += elapsed_time
        cnt += 1

        # save to file
        # s_tmp = '%s/out/%s' % (save_dir)
        # folder = s_tmp[:s_tmp.rfind('/')]
        # if not os.path.isdir(folder):
        #     print('Make a new folder: ', folder)
        #     mkdir_p(folder)
        # random = str(time.time())
        # fullpath = '%s_%2d_%r.png' % (s_tmp,j,random)
        # im.save(fullpath)

        # return
        return im

    print("Average time: %.2f" % (time_sum / cnt))


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        #assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        #assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    if torch.cuda.is_available():
        text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))

    # dream endpoint
    @app.route('/<dream>')
    def serve_img(dream):
        #dream = request.args.get('dream', default = '', type = str)
        if len(dream.strip()) == 0:
            abort(Response("Please input a dream"))
        # else:
        #     im = sampling(dream, text_encoder, netG, dataloader,device)
        #     return serve_pil_image(im)
        else:
            print("This server is running the following OS: ", platform.platform(), "Please adjust the 'video_codec' and 'video_container' variable according to the OS. See this link for options: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html")
            im_list = sampling(dream, text_encoder, netG, dataloader,device)
            print('im_list: ', im_list)
            save_pil_imgs_to_disk(im_list)
            return "Images generated. See: "+UPLOAD_FOLDER

    # health check
    @app.route('/')
    def hello():
        return 'Hello, World!'

    # start server
    if args.ssl_cert and args.ssl_key:
        ssl_ctx = (args.ssl_cert, args.ssl_key)
        print(ssl_ctx)
        app.run(host="0.0.0.0", port=args.port, ssl_context=ssl_ctx)
    else:
        app.run(host="0.0.0.0", port=args.port)
