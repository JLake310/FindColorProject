import sys
import io
import os
import argparse
import numpy as np
import segmentation_models_pytorch as smp
import datetime
import glob

from PyQt5.QtGui import QColor, QPen, QImage, QPainter, QIcon
from PyQt5.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QPoint, QPointF
from timm.models import create_model
from flask import Flask, render_template, request, url_for, redirect
from flaskwebgui import FlaskUI
# from flaskwebgui import FlaskUI
import webbrowser as wb
from importlib import import_module
from flask_restx import Resource, Api
from skimage import color

import base64
# from threading import Thread
import socket
import torch
#
from io import BytesIO
import cv2
from PIL import Image
import base64
from base64 import b64encode
from gui import gui_draw
from gui import gui_main
import modeling
from einops import rearrange
from gui.lab_gamut import abGrid, lab2rgb_1d, rgb2lab_1d
from gui.lab_gamut import snap_ab

used_colors = []

global mode
mode = "ours"


def mode_info(md):
    global mode
    mode = md
    print(f"colorization mode selected: {mode}")
    return mode


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        mask_cent=args.mask_cent,
    )
    return model


def get_args(filepath):
    parser = argparse.ArgumentParser('Colorization UI', add_help=False)
    # Directories
    parser.add_argument('--model_path', type=str, default='path/to/checkpoints', help='checkpoint path of model')
    parser.add_argument('--target_image', default=filepath, type=str, help='validation dataset path')
    parser.add_argument('--device', default='cpu', help='device to use for testing')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')

    # Model parameters
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode')
    parser.add_argument('--mask_cent', action='store_true', help='mask_cent')

    args = parser.parse_args()

    return args


class GUIPalette():
    def __init__(self, grid_sz=(6, 3)):
        self.color_width = 25
        self.border = 6
        self.win_width = grid_sz[0] * self.color_width + (grid_sz[0] + 1) * self.border
        self.win_height = grid_sz[1] * self.color_width + (grid_sz[1] + 1) * self.border
        self.setFixedSize(self.win_width, self.win_height)
        self.num_colors = grid_sz[0] * grid_sz[1]
        self.grid_sz = grid_sz
        self.colors = None
        self.color_id = -1
        self.reset()

    def set_colors(self, colors):
        if colors is not None:
            self.colors = (colors[:min(colors.shape[0], self.num_colors), :] * 255).astype(np.uint8)
            self.color_id = -1
            # self.update()

    def paintEvent(self, event):
        # painter = QPainter()
        # painter.begin(self)
        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.fillRect(event.rect(), Qt.white)
        if self.colors is not None:
            for n, c in enumerate(self.colors):
                # ca = QColor(c[0], c[1], c[2], 255)
                # painter.setPen(QPen(Qt.black, 1))
                # painter.setBrush(ca)
                grid_x = n % self.grid_sz[0]
                grid_y = (n - grid_x) // self.grid_sz[0]
                x = grid_x * (self.color_width + self.border) + self.border
                y = grid_y * (self.color_width + self.border) + self.border

                if n == self.color_id:
                    painter.drawEllipse(x, y, self.color_width, self.color_width)
                else:
                    painter.drawRoundedRect(x, y, self.color_width, self.color_width, 2, 2)

        painter.end()

    def sizeHint(self):
        return QSize(self.win_width, self.win_height)

    def reset(self):
        self.colors = None
        self.mouseClicked = False
        self.color_id = -1
        self.update()

    def selected_color(self, pos):
        width = self.color_width + self.border
        dx = pos.x() % width
        dy = pos.y() % width
        if dx >= self.border and dy >= self.border:
            x_id = (pos.x() - dx) // width
            y_id = (pos.y() - dy) // width
            color_id = x_id + y_id * self.grid_sz[0]
            return int(color_id)
        else:
            return -1

    def update_ui(self, color_id):
        self.color_id = int(color_id)
        self.update()
        if color_id >= 0 and self.colors is not None:
            print('choose color (%d) type (%s)' % (color_id, type(color_id)))
            color = self.colors[color_id]
            # self.emit(SIGNAL('update_color'), color)
            self.update_color.emit(color)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # click the point
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)
            self.mouseClicked = True

    def mouseMoveEvent(self, event):
        if self.mouseClicked:
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False


class GUIGamut():
    def __init__(self, gamut_size=110):
        self.gamut_size = gamut_size
        self.win_size = gamut_size * 2  # divided by 4
        # self.setFixedSize(self.win_size, self.win_size)
        self.ab_grid = abGrid(gamut_size=gamut_size, D=1)
        self.reset()
        # self.update()

    def set_gamut(self, l_in=50):
        if len(L_arr) > 1:
            self.l_in = L_arr[-1]
            self.ab_map, self.mask = self.ab_grid.update_gamut(l_in=self.l_in)
        else:
            self.l_in = l_in
            self.ab_map, self.mask = self.ab_grid.update_gamut(l_in=self.l_in)
        # self.update()

    def set_ab(self, color):
        #  팔레트 색깔인가??
        self.color = color
        self.lab = rgb2lab_1d(self.color)
        x, y = self.ab_grid.ab2xy(self.lab[1], self.lab[2])
        pos_arr.append([x, y])
        # self.pos = QPointF(x, y)
        # self.update()

    def is_valid_point(self, pos):
        if pos is None or self.mask is None:
            return False
        else:
            x = pos.x()
            y = pos.y()
            if x >= 0 and y >= 0 and x < self.win_size and y < self.win_size:
                return self.mask[y, x]
            else:
                return False

    def update_ui(self, pos):
        self.pos = pos
        a, b = self.ab_grid.xy2ab(pos[0], pos[1])
        # get color we need L
        L = self.l_in
        lab = np.array([L, a, b])
        color = lab2rgb_1d(lab, clip=True, dtype='uint8')
        print('update_ui_color:', color)
        # self.emit(SIGNAL('update_color'), color)
        self.color = color
        crg_arr.append(color)
        # self.update_color.emit(color)
        # self.update()

    def paintEvent(self):
        # painter = QPainter()
        # painter.begin(self)
        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.fillRect(event.rect(), Qt.white)
        if self.ab_map is not None:
            ab_map = cv2.resize(self.ab_map, (self.win_size, self.win_size))
            pil_image = Image.fromarray(ab_map)
            img_str = cv2.imencode('.png', ab_map)[1].tostring()

            image_io = BytesIO(img_str)
            pil_image.save(image_io, 'PNG')
            dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')

            return dataurl

        # painter.drawLine 알아봐야 함

        # if self.pos is not None:
        # painter.setPen(QPen(Qt.black, 2, Qt.SolidLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
        #     w = 5
        #     x = self.pos.x()
        #     y = self.pos.y()
        #     painter.drawLine(x - w, y, x + w, y)
        #     painter.drawLine(x, y - w, x, y + w)
        # painter.end()

    def mousePressEvent(self, event):
        pos = event.pos()
        if event.button() == Qt.LeftButton and self.is_valid_point(pos):  # click the point
            self.update_ui(pos)
            self.mouseClicked = True

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self.is_valid_point(pos):
            if self.mouseClicked:
                self.update_ui(pos)

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)

    def reset(self):
        self.ab_map = None
        self.mask = None
        self.color = None
        self.lab = None
        self.pos = None
        self.mouseClicked = False
        # self.update()


class UserEdit(object):
    def __init__(self, mode, win_size, load_size, img_size):
        self.mode = mode
        self.win_size = win_size
        self.img_size = img_size
        self.load_size = load_size
        print('image_size', self.img_size)
        max_width = np.max(self.img_size)
        self.scale = float(max_width) / self.load_size  # original image to 224 ration
        self.dw = int((self.win_size - img_size[0]) // 2)
        self.dh = int((self.win_size - img_size[1]) // 2)
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.ui_count = 0
        print(self)

    def scale_point(self, in_x, in_y, w):
        x = int((in_x - self.dw) / float(self.img_w) * self.load_size) + w
        y = int((in_y - self.dh) / float(self.img_h) * self.load_size) + w
        return x, y

    def __str__(self):
        return "add (%s) with win_size %3.3f, load_size %3.3f" % (self.mode, self.win_size, self.load_size)


class PointEdit(UserEdit):
    def __init__(self, win_size, load_size, img_size):
        UserEdit.__init__(self, 'point', win_size, load_size, img_size)

    def add(self, pnt, color, userColor, width, ui_count):
        self.pnt = pnt
        self.color = color
        self.userColor = userColor
        self.width = width
        self.ui_count = ui_count

    def select_old(self, pnt, ui_count):
        self.pnt = pnt
        self.ui_count = ui_count
        return self.userColor, self.width

    def update_color(self, color, userColor):
        self.color = color
        self.userColor = userColor

    def updateInput(self, im, mask, vis_im):
        w = int(self.width / self.scale)
        pnt = self.pnt
        x1, y1 = self.scale_point(pnt[0], pnt[1], -w)
        tl = (x1, y1)
        # x2, y2 = self.scale_point(pnt.x(), pnt.y(), w)
        # br = (x2, y2)
        br = (x1 + 1, y1 + 1)  # hint size fixed to 2
        c = (self.color.red(), self.color.green(), self.color.blue())
        uc = (self.userColor.red(), self.userColor.green(), self.userColor.blue())
        cv2.rectangle(mask, tl, br, 0, -1)
        cv2.rectangle(im, tl, br, c, -1)
        cv2.rectangle(vis_im, tl, br, uc, -1)

    def is_same(self, pnt):
        dx = abs(self.pnt[0] - pnt[0])
        dy = abs(self.pnt[1] - pnt[1])
        return dx <= self.width + 1 and dy <= self.width + 1

    def update_painter(self, painter):
        w = max(3, self.width)
        c = self.color
        r = c.red()
        g = c.green()
        b = c.blue()
        ca = QColor(c.red(), c.green(), c.blue(), 255)
        d_to_black = r * r + g * g + b * b
        d_to_white = (255 - r) * (255 - r) + (255 - g) * (255 - g) + (255 - r) * (255 - r)
        if d_to_black > d_to_white:
            painter.setPen(QPen(Qt.black, 1))
        else:
            painter.setPen(QPen(Qt.white, 1))
        painter.setBrush(ca)
        painter.drawRoundedRect(self.pnt.x() - w, self.pnt.y() - w, 1 + 2 * w, 1 + 2 * w, 2, 2)


class UIControl:
    def __init__(self, win_size=256, load_size=224):
        self.win_size = win_size
        self.load_size = load_size
        self.reset()
        self.userEdit = None
        self.userEdits = []
        self.ui_count = 0

    def setImageSize(self, img_size):
        self.img_size = img_size

    def addStroke(self, prevPnt, nextPnt, color, userColor, width):
        pass

    def erasePoint(self, pnt):
        isErase = False
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdits.remove(ue)
                print('remove user edit %d\n' % id)
                isErase = True
                break
        return isErase

    def addPoint(self, pnt, color, userColor, width):
        self.ui_count += 1
        print('process add Point')
        self.userEdit = None
        isNew = True
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdit = ue
                isNew = False
                print('select user edit %d\n' % id)
                break

        if self.userEdit is None:
            self.userEdit = PointEdit(self.win_size, self.load_size, self.img_size)
            self.userEdits.append(self.userEdit)
            print('|addPoint| add user edit %d\n' % len(self.userEdits))
            self.userEdit.add(pnt, color, userColor, width, self.ui_count)
            return userColor, width, isNew
        else:
            userColor, width = self.userEdit.select_old(pnt, self.ui_count)
            return userColor, width, isNew

    def movePoint(self, pnt, color, userColor, width):
        self.userEdit.add(pnt, color, userColor, width, self.ui_count)

    def update_color(self, color, userColor):
        self.userEdit.update_color(color, userColor)

    def update_painter(self, painter):
        for ue in self.userEdits:
            if ue is not None:
                ue.update_painter(painter)

    def get_stroke_image(self, im):
        return im

    def used_colors(self):  # get recently used colors
        if len(self.userEdits) == 0:
            return None
        nEdits = len(self.userEdits)
        ui_counts = np.zeros(nEdits)
        ui_colors = np.zeros((nEdits, 3))
        for n, ue in enumerate(self.userEdits):
            ui_counts[n] = ue.ui_count
            c = ue.userColor
            ui_colors[n, :] = [c.red(), c.green(), c.blue()]

        ui_counts = np.array(ui_counts)
        ids = np.argsort(-ui_counts)
        ui_colors = ui_colors[ids, :]
        unique_colors = []
        for ui_color in ui_colors:
            is_exit = False
            for u_color in unique_colors:
                d = np.sum(np.abs(u_color - ui_color))
                if d < 0.1:
                    is_exit = True
                    break

            if not is_exit:
                unique_colors.append(ui_color)

        unique_colors = np.vstack(unique_colors)
        return unique_colors / 255.0

    def get_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)
        vis_im = np.zeros((h, w, 3), np.uint8)
        print('get input', self.userEdits)
        for ue in self.userEdits:
            ue.updateInput(im, mask, vis_im)

        return im, mask

    def reset(self):
        self.userEdits = []
        self.userEdit = None
        self.ui_count = 0


class GUI_VIS():
    def __init__(self, win_size=256, scale=2.0):
        self.result = None
        self.win_width = win_size
        self.win_height = win_size
        self.scale = scale

    # def paintEvent(self, event):
    #     painter = QPainter()
    #     painter.begin(self)
    #     painter.setRenderHint(QPainter.Antialiasing)
    #     painter.fillRect(event.rect(), QColor(49, 54, 49))
    #     if self.result is not None:
    #         h, w, c = self.result.shape
    #         qImg = QImage(self.result.tostring(), w, h, QImage.Format_RGB888)
    #         dw = int((self.win_width - w) // 2)
    #         dh = int((self.win_height - h) // 2)
    #         painter.drawImage(dw, dh, qImg)

    def update_result(self, result):
        self.result = result
        self.update()

    def scale_point(self, pnt):
        x = int(pnt[0] / scale_arr[-1])
        y = int(pnt[1] / scale_arr[-1])
        return x, y

    def mousePressEvent(self, pos):
        x, y = self.scale_point(pos)
        if result_arr[-1] is not None:
            color = result_arr[-1][y, x, :]  #
            print('color', color)


class GUIDraw():
    # Signals
    update_color = pyqtSignal(str)
    update_gammut = pyqtSignal(object)
    used_colors = pyqtSignal(object)
    update_ab = pyqtSignal(object)
    update_result = pyqtSignal(object)

    def __init__(self, model=None, nohint_model=None, load_size=224, win_size=512, device='cpu'):
        self.image_file = None
        self.pos = None
        self.model = model
        # add
        self.nohint_model = nohint_model

        self.win_size = win_size
        self.load_size = load_size
        self.device = device
        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.movie = True
        self.init_color()  # initialize color
        self.im_gray3 = None
        self.eraseMode = False
        self.ui_mode = 'none'  # stroke or point
        self.image_loaded = False
        self.use_gray = True
        self.total_images = 0
        self.image_id = 0

    def clock_count(self):
        self.count_secs -= 1
        self.update()

    def init_result(self, image_file):
        # self.read_image(image_file.encode('utf-8'))  # read an image
        self.read_image(image_file)  # read an image
        ##############################
        # my model
        im_full = cv2.resize(self.im_full, (768, 768), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
        gray = np.stack([gray, gray, gray], -1)
        l_img = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)[:, :, [0]].transpose((2, 0, 1))
        l_img = torch.from_numpy(l_img).type(torch.FloatTensor).to(self.device) / 255
        ab = self.nohint_model(l_img.unsqueeze(0))[0]  # .detach().cpu().numpy().transpose((1,2,0))

        lab = torch.cat([l_img, ab], axis=0).permute(1, 2, 0).cpu().detach().numpy() * 255  # h,w,c
        lab = lab.astype(np.uint8)
        self.my_results = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # 왜.. gbr밖에

        #######
        # 저장용
        ab = ab.permute(1, 2, 0).cpu().detach().numpy() * 255
        ab = cv2.resize(ab, (self.im_full.shape[1], self.im_full.shape[0]), interpolation=cv2.INTER_AREA)  # INTER_CUBIC
        im_l = cv2.cvtColor(self.im_full, cv2.COLOR_BGR2LAB)[:, :, [0]]
        org_my_results = np.concatenate([im_l, ab], axis=2)
        org_my_results = org_my_results.astype(np.uint8)

        self.org_my_results = cv2.cvtColor(org_my_results, cv2.COLOR_LAB2BGR)  # 왜.. gbr밖에

        ##############################
        #
        # self.reset()

    def get_batches(self, img_dir):
        self.img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
        self.total_images = len(self.img_list)
        img_first = self.img_list[0]
        self.init_result(img_first)

    def get_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)
        vis_im = np.zeros((h, w, 3), np.uint8)

        for ue in self.userEdits:
            ue.updateInput(im, mask, vis_im)

        return im, mask

    def valid_point(self, pnt):
        print('gui_draw valid_point', pnt)
        if pnt is None:
            print('WARNING: no point\n')
            return None
        else:
            if pnt[0] >= self.dw and pnt.y() >= self.dh and pnt[0] < self.win_size - self.dw and pnt[
                1] < self.win_size - self.dh:
                x = int(np.round(pnt[0]))
                y = int(np.round(pnt[1]))
                return x, y
            else:
                print('WARNING: invalid point (%d, %d)\n' % (pnt[0], pnt[1]))
                return None

    def nextImage(self):
        self.save_result()
        self.image_id += 1
        if self.image_id == self.total_images:
            print('you have finished all the results')
            sys.exit()
        img_current = self.img_list[self.image_id]
        # self.reset()
        self.init_result(img_current)
        self.reset_timer()

    def read_image(self, image_file):
        # self.result = None
        self.image_loaded = True
        self.image_file = image_file
        # print(image_file)
        im_bgr = cv2.imread(self.image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        scale_arr.append(float(self.win_size) / self.load_size)
        self.scale = float(self.win_size) / self.load_size
        print('scale = %f' % self.scale)
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh
        self.uiControl.setImageSize((rw, rh))
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])

        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

    def compute_result(self, model, my_model):
        self.model = model
        self.nohint_model = my_model
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))  # (1, H, W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))  # (3, H, W)
        self.im_ab0 = im_lab[1:3, :, :]

        # _im_lab is 1) normalized 2) a torch tensor
        _im_lab = self.im_lab.transpose((2, 0, 1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :] - 50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)

        # _img_mask is 1) normalized ab 2) flipped mask
        _img_mask = np.concatenate((self.im_ab0 / 110, (255 - self.im_mask0) / 255), axis=0)
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device)

        # _im_lab is the full color image, _img_mask is the ab_hint+mask
        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                       h=self.load_size // self.model.patch_size, w=self.load_size // self.model.patch_size,
                       p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().numpy()
        self.ab = ab

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA)  # INTER_CUBIC
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        #########
        # my model
        #########
        my_results = cv2.resize(self.my_results, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA).astype(
            np.float32)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255.)
        # pred_rgb = my_results

        print(f"current mode {mode}")
        if mode == "ours":
            pred_rgb = my_results * 0.5 + pred_rgb * 0.5
        elif mode == "nohint":
            pred_rgb = my_results

        pred_rgb = pred_rgb.astype('uint8')
        #####################################################
        self.result = pred_rgb
        # self.emit(SIGNAL('update_result'), self.result)
        result_arr.append(self.result)
        return self.result
        # self.emit(SIGNAL('update_result'), self.result)
        # self.update()

    def reset(self):
        self.ui_mode = 'none'
        self.pos = None
        self.result = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        # self.update_result.emit(None)
        # self.update()

    def save_result(self):
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)

        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = "_".join([path, suffix])

        print('saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)

    # gamut 연관있는 함수
    def mousePressEvent(self, event):
        print('mouse press', event.pos())
        pos = self.valid_point(event.pos())

        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)
                self.update_ui(move_point=False)
                self.compute_result()

            if event.button() == Qt.RightButton:
                # draw the stroke
                self.pos = pos
                self.ui_mode = 'erase'
                self.update_ui(move_point=False)
                self.compute_result()

    def mouseMoveEvent(self, event):
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.ui_mode == 'point':
                self.update_ui(move_point=True)
                self.compute_result()

    def paintEvent(self):
        # painter = QPainter()
        # painter.begin(self)
        # painter.fillRect(event.rect(), QColor(49, 54, 49))
        # painter.setRenderHint(QPainter.Antialiasing)
        if self.use_gray or self.result is None:
            im = self.gray_win
        else:
            im = self.result

        if im is not None:
            print(im)
            # qImg = QImage(im.tostring(), im.shape[1], im.shape[0], QImage.Format_RGB888)
            # painter.drawImage(self.dw, self.dh, qImg)

        # self.uiControl.update_painter(painter)
        # painter.end()

    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()

    def erase(self):
        self.eraseMode = not self.eraseMode

    def set_color(self, c_rgb):
        c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, pos2_arr[-1])
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        # self.update_color.emit(str('background-color: %s' % self.color.name()))
        print("set snap: ", snap_qcolor, "set user: ", self.user_color)
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_changed_color()

    def compute_changed_color(self):
        print('gui_draw compute_changed_color')

        im, mask = self.uiControl.get_input()
        # print("computed의 im,mask: ", im[112:], len[112:])
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))  # (1, H, W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))  # (3, H, W)
        self.im_ab0 = im_lab[1:3, :, :]

        # _im_lab is 1) normalized 2) a torch tensor
        _im_lab = self.im_lab.transpose((2, 0, 1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :] - 50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)

        # _img_mask is 1) normalized ab 2) flipped mask
        _img_mask = np.concatenate((self.im_ab0 / 110, (255 - self.im_mask0) / 255), axis=0)
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device)

        # _im_lab is the full color image, _img_mask is the ab_hint+mask
        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                       h=self.load_size // self.model.patch_size, w=self.load_size // self.model.patch_size,
                       p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().numpy()

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb

        result_arr.append(self.result)
        return self.result
        # self.emit(SIGNAL('update_result'), self.result)
        # self.update()

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)
        print('gui_draw calibrate_color', c)
        print('calibrate_color', c)
        # snap color based on L color
        color_array = np.array([128, 128, 128]).astype('uint8')
        mean_L = self.im_l[y, x]
        print('calibrate_color array', color_array)
        print('calibrate_color mean_L', mean_L)
        snap_color = snap_ab(mean_L, color_array)

        snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def change_color(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            L = self.im_lab[y, x, 0]
            L_arr.append(L)

            #  set gamut으로 연결
            # self.update_gammut.emit(L)

            # used_colors = self.uiControl.used_colors()
            # self.emit(SIGNAL('used_colors'), used_colors)

            # used_colors.append(used_colors) nonetype 비교해볼것
            snap_color = self.calibrate_color(self.user_color, pos)
            c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)
            C_arr.append(c)
            #  set ab로 연결
            # self.emit(SIGNAL('update_ab'), c)
            # self.update_ab.emit(c)

    def init_color(self):
        self.user_color = np.array([128, 128, 128], dtype=np.uint8)  # default color red
        self.color = self.user_color

    def scale_point(self, pnt):
        x = int((pnt[0] - self.dw) / self.win_w * self.load_size)
        y = int((pnt[1] - self.dh) / self.win_h * self.load_size)
        return x, y

    def update_ui(self, pos):
        # if self.ui_mode == 'none':
        #     return False
        is_predict = False
        print("update_ui", self.brushWidth)
        snap_qcolor = self.calibrate_color(self.user_color, pos)
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        # self.update_color.emit(str('background-color: %s' % self.color.name()))

        # if self.ui_mode == 'point':
        #     if move_point:
        #         self.uiControl.movePoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
        #     else:
        # self.user_color 뒤에 걸 snap_qcolor로 바꿔서 넣어봄
        self.user_color, self.brushWidth, isNew = self.uiControl.addPoint(pos, snap_qcolor, self.user_color,
                                                                          self.brushWidth)
        # if isNew:
        #     is_predict = True
        #     # self.predict_color()

        is_predict = True
        # if self.ui_mode == 'stroke':
        #     self.uiControl.addStroke(self.prev_pos, self.pos, snap_qcolor, self.user_color, self.brushWidth)
        # if self.ui_mode == 'erase':
        #     isRemoved = self.uiControl.erasePoint(self.pos)
        #     if isRemoved:
        #         is_predict = True
        #         # self.predict_color()
        return is_predict


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    app = Flask(__name__)

    G = GUIDraw()
    Gamut = GUIGamut()
    mem_bytes = io.BytesIO()
    self_imlab = []

    gamut_img_arr = []
    img_src_arr = []
    uri_arr = []
    pos_arr = []
    pos2_arr = []

    self_lin = []

    image_src = []
    scale_arr = []
    result_arr = []
    L_arr = []
    C_arr = []

    dw_arr = []
    dh_arr = []

    crg_arr = []


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/down')
def down():
    return render_template('down.html')


@app.route('/index')
def index():
    return render_template('index.html', filename=img_src_arr, result=uri_arr, id="result", label="result!!")


@app.route('/upload', methods=['post'])
def upload():
    cpath = os.getcwd()
    print(cpath)
    os.chdir(cpath)
    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename
        img_src = url_for('static', filename='image/' + filename)
        image_file = cpath + '/static/image/' + filename
        print(image_file)
        file.save(os.path.join('./static/image', filename))
        img_src_arr.append(img_src)
        image_src.append(image_file)

    image_file = image_src[-1]
    img_src = img_src_arr[-1]

    args = get_args(image_file)
    model = get_model(args)
    model.to(args.device)
    checkpoint = torch.load('./icolorit_base_4ch_patch16_224.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # no hint model
    print('Creating model:', 'nohint model')
    nohint_model = smp.Unet(
        encoder_name='efficientnet-b1',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your dataset)
        activation='sigmoid',
    )
    # 쓸모없는 키 이름 지우기
    from collections import OrderedDict
    temp_dict = OrderedDict()
    weight = torch.load('./fold0_best_e49.pth', map_location='cpu')
    for k, w in weight.items():
        temp_dict[k.replace('module.model.', '')] = w

    #
    nohint_model.load_state_dict(temp_dict)
    nohint_model.eval()

    G.nohint_model = nohint_model
    G.init_result(image_file)
    img = G.compute_result(model, nohint_model)
    img = Image.fromarray(img.astype("uint8"))
    img.save(mem_bytes, 'JPEG')
    mem_bytes.seek(0)
    img_base64 = base64.b64encode(mem_bytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s" % (mime, img_base64)
    uri_arr.append(uri)

    # Gamut.set_gamut()
    # gamut_img = Gamut.paintEvent()
    # gamut_img_arr.append(gamut_img)

    return redirect(url_for('index'))


@app.route('/download', methods=['post'])
def download():
    G.save_result()
    return redirect(url_for('index'))


@app.route('/restart', methods=['post'])
def restart():
    temp1 = img_src_arr[0]
    temp2 = image_src[0]
    img_src_arr.clear()
    image_src.clear()
    img_src_arr.append(temp1)
    image_src.append(temp2)

    gamut_img_arr.clear()
    uri_arr.clear()
    pos_arr.clear()
    scale_arr.clear()
    result_arr.clear()
    L_arr.clear()
    C_arr.clear()
    return redirect(url_for('index'))


@app.route('/quit', methods=['post'])
def quit():
    uri_arr.clear()
    pos_arr.clear()
    gamut_img_arr.clear()
    img_src_arr.clear()
    image_src.clear()
    scale_arr.clear()
    result_arr.clear()
    L_arr.clear()
    C_arr.clear()
    return redirect(url_for('index'))


@app.route('/pos', methods=['post'])
def pos():
    vis = GUI_VIS()
    x_pos = int(request.form['x_pos'])
    y_pos = int(request.form['y_pos'])
    pos_arr.append([x_pos, y_pos])

    vis.mousePressEvent(pos_arr[-1])

    G.init_color()
    G.init_result(image_src[-1])
    G.change_color(pos_arr[-1])

    Gamut.set_gamut()
    Gamut.set_ab(C_arr[-1])
    Gamut.update_ui(pos_arr[-1])
    gamut_img = Gamut.paintEvent()
    gamut_img_arr.append(gamut_img)
    return redirect(url_for('index'))


@app.route('/pos2', methods=['post'])
def pos2():
    x_pos = int(request.form['x_pos2'])
    y_pos = int(request.form['y_pos2'])
    print(x_pos, y_pos)
    pos2_arr.append([x_pos, y_pos])
    # update ui를 통해서 color 값을 산출 draw의 color set으로 값 이동 후 compute recolorization 실행!
    Gamut.update_ui(pos2_arr[-1])

    G.update_ui(pos2_arr[-1])
    G.set_color(crg_arr[-1])

    # G.init_color()
    # G.change_color(pos2_arr[-1])

    print(np.array_equal(result_arr[-2], result_arr[-1]))

    img = result_arr[-1]
    img = Image.fromarray(img.astype("uint8"))
    img.save(mem_bytes, 'JPEG')
    mem_bytes.seek(0)
    img_base64 = base64.b64encode(mem_bytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s" % (mime, img_base64)
    uri_arr.append(uri)
    G.save_result()

    # Gamut.update_ui(pos2_arr[-1])
    # G.set_color(gamut_color_arr[-1])
    # G.change_color(pos_arr[-1])

    return redirect(url_for('index'))


app.run()

