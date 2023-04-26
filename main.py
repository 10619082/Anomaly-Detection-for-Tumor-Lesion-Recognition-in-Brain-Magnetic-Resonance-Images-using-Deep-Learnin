import itertools

import skimage

import random

save_ensemble = True
save_statistic = True
save_horizontally = True
save_heatmap = True

# se è ture allora utilizzo il mad
MAD_on = True

import scipy.stats as st

# se è true allora esco da compute statistic appena calcolato le binary maps delle slice di tutti pazienti
short_statistic = False

# salto direttamento al filtraggio della maschera da fare soltanto quando ho giè ensemble mask calcolati
short_ensemble = True

from sklearn.ensemble import IsolationForest

import math

import glob
import matplotlib
import seaborn as sns

from patchify import patchify

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from scipy.stats import shapiro
from scipy.stats import normaltest

import pandas as pd

from PIL import Image

to_pil_image = transforms.ToPILImage()
from itertools import combinations

from tqdm import tqdm

from matplotlib.image import imread
import matplotlib.pyplot as plt

from data.BrainMetShareDataset import *
from data.DatasetGenerator import *
from data.utils import *
import numpy
from statsmodels.graphics.gofplots import qqplot

from src.data.dataset import get_training_dataloader, get_test_dataloader, get_validation_dataloader
from src.models.VQVAE.VQVAEModel import VQVAEModel
import os

from numpy.random import seed
from numpy.random import randn

torch.manual_seed(0)
"""CUDA"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dice_list = []
first_list = []
second_list = []


# crea una figura di immagini che mostrano i risultati ottenuti dall'ensemble
# l'immagine creata mostrerà: imm originale, segmentazione del radiologo, anomaly map generata
def showImagesHorizontally_ensemble(list_of_files, path):
    # ci metto anche originale
    fig = matplotlib.pyplot.figure()
    # 1 orig 2 seg 3 anomaly
    fig.set_size_inches(50, 10)

    dice, first, second = new_dice(list_of_files[1], np.array(list_of_files[2])[:, :, 1] / 255)
    dice_list.append(dice)
    first_list.append(first)
    second_list.append(second)

    recall, precision, f1_score, FNR, FPR = define_metrics(list_of_files[1], np.array(list_of_files[2])[:, :, 1] / 255)
    '''print('Dice : {:.3f}'.format(dice) + '   First factor: {:.3f}'.format(first) + '   Second factor: {:.3f}'.format(
            second) +
        '   Recall: {:.3f}'.format(recall) + '   Precision: {:.3f}'.format(precision) + '   F1_score: {:.3f}'.format(
            f1_score) +
        '   FNR: {:.3f}'.format(FNR) + '   FPR: {:.3f}'.format(FPR) )'''
    fig.suptitle(
        'Dice : {:.3f}'.format(dice) + '   First factor: {:.3f}'.format(first) + '   Second factor: {:.3f}'.format(
            second) +
        '   Recall: {:.3f}'.format(recall) + '   Precision: {:.3f}'.format(precision) + '   F1_score: {:.3f}'.format(
            f1_score) +
        '   FNR: {:.3f}'.format(FNR) + '   FPR: {:.3f}'.format(FPR), fontsize=30)

    # matplotlib.pyplot.xlabel('Dice : ' + str(dice) + ' First factor: ' + str(first) + ' Second factor ' + str(second), fontsize=15)

    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        # print('i')
        a = fig.add_subplot(1, number_of_files, i + 1)
        image = list_of_files[i]
        if i == 0 or i == 1:
            image = image * 255
        if i == 2:
            matplotlib.pyplot.imshow(image)
        else:
            matplotlib.pyplot.imshow(image, cmap='Greys_r')
        matplotlib.pyplot.axis('off')
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig(path)


def qualtile(data):
    # q-q plot
    qqplot(np.array(data), line='s')
    matplotlib.pyplot.show()


def get_angle(p1, p2):
    """Get the angle of this line with the horizontal axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle


def normality_test_shapiro(data):
    # normality test
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


def plot_histogram(data):
    matplotlib.pyplot.hist(data, bins=400)
    matplotlib.pyplot.xlabel("pixel values")
    matplotlib.pyplot.ylabel("relative frequency")
    matplotlib.pyplot.title("Number of slices: " + str(len(data)))
    matplotlib.pyplot.xlim([-50, 50])
    matplotlib.pyplot.show()


def Cloning(li1):
    li_copy = []
    li_copy.extend(li1)
    return li_copy


# come funzione showImagesHorizontally_ensemble soltanto che l'immagine in output prodotta
# è composta da: imm originale, imm ricostruita, imm diff in heatmap, segmentazione radiolo, anomaly map generata
def showImagesHorizontally(list_of_files, path):
    fig = matplotlib.pyplot.figure()

    fig.set_size_inches(50, 10)

    dice, first, second = new_dice(list_of_files[3], np.array(list_of_files[4]))
    recall, precision, f1_score, FNR, FPR = define_metrics(list_of_files[3], np.array(list_of_files[4]))

    fig.suptitle(
        'Dice : {:.3f}'.format(dice) + '   First factor: {:.3f}'.format(first) + '   Second factor: {:.3f}'.format(
            second) +
        '   Recall: {:.3f}'.format(recall) + '   Precision: {:.3f}'.format(precision) + '   F1_score: {:.3f}'.format(
            f1_score) +
        '   FNR: {:.3f}'.format(FNR) + '   FPR: {:.3f}'.format(FPR), fontsize=30)
    # matplotlib.pyplot.xlabel('Dice : ' + str(dice) + ' First factor: ' + str(first) + ' Second factor ' + str(second), fontsize=15)

    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        if i == 5: break
        a = fig.add_subplot(1, number_of_files, i + 1)
        image = list_of_files[i]
        if i == 2:
            sns.heatmap(image, cmap="coolwarm", cbar=True)
        if i == 4:
            # list_of_files[4] è anomaly map,
            # ist_of_files[3] è seg
            # list_of_files[4] è yellow pixels
            image = three_channels_AM(list_of_files[4], np.array(list_of_files[3]), np.array(list_of_files[5]))
            matplotlib.pyplot.imshow(image)
        else:
            matplotlib.pyplot.imshow(image, cmap='Greys_r')
        matplotlib.pyplot.axis('off')
    # matplotlib.pyplot.show()
    if save_horizontally: matplotlib.pyplot.savefig(path)


def show_RGB(img):
    Image.fromarray((np.uint8(img)), 'RGB').show()


def show_grey(img):
    img = np.squeeze(np.array(img))
    if np.max(img) <= 1:
        img *= 255
    Image.fromarray(np.uint8(img)).show()


def check_std(std, anomaly_map):
    '''assert std.shape == anomaly_map.shape

    log_std = np.asarray(std).astype(bool)
    log_map = np.asarray(anomaly_map).astype(bool)

    intersection = np.logical_and(log_std, log_map)
    blue_pixels = log_map ^ log_std

    blue_pixels = blue_pixels ^ log_std


    return intersection*1, blue_pixels*1'''

    intersection = std * anomaly_map
    blue_pixels = np.array(anomaly_map) - np.array(std)
    blue_pixels[blue_pixels < 0] = 0
    return intersection, blue_pixels


def new_dice(true_mask, pred_mask):
    pred_mask[pred_mask > 0] = 1
    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)

    a = pred_mask.sum()

    intersection = np.logical_and(true_mask, pred_mask)

    b = intersection.sum()

    # ^ è bitwise xor sarebbe il meno
    anomaly_pixel_outside_the_mask = pred_mask ^ intersection
    anomaly_pixel_outside_the_mask = anomaly_pixel_outside_the_mask.sum()
    pred_mask = pred_mask.sum()
    true_mask = true_mask.sum()
    intersection = intersection.sum()

    # pixel anomali accesi dentro la maschera dati dal modello/pixel della maschera
    dice_score = intersection / true_mask

    if pred_mask != 0:
        false_positive = anomaly_pixel_outside_the_mask / pred_mask  # diviso  ( # di pixel area di consenso ensemble )
        factor = 1 - false_positive
    else:
        # print('No anomaly detected prediction = 0')
        factor = 0

    # if dice_score == 0:  print('No anomaly detected intersection between true mask and pred mask ')

    loss = math.sqrt(dice_score * factor)
    # (dice_score^a*factor^b)^(1/a+b) se voglio dare più peso a factor impongo b > a
    # togli prima immagine dal test set

    return loss, dice_score, factor


# riceve in input una imm e in output da due immagini: l'input e l'input con una parte di contorno eliminata
def create_images_masks(img):
    # img deve essere np array (256,256)
    ret, full_image_mask = cv2.threshold(np.array(img), 0, 255, cv2.THRESH_BINARY)

    # visualize full image mask
    # Image.fromarray(np.uint8(full_image_mask)).show()

    full_image_mask = full_image_mask.astype(np.uint8)

    contours, hierarchy = cv2.findContours(full_image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find countour

    # contours = contours.all()

    obj_index = contours.index(max(contours, key=len))  # find index of largest object

    resized_image_mask = cv2.drawContours(np.array(full_image_mask), contours, obj_index, (0),
                                          3)  # draw coutour on original image

    # plot the new mask decreased of size
    # Image.fromarray(np.uint8(resized_image_mask)).show()

    # plot the residual
    # Image.fromarray(np.uint8(np.subtract(full_image_mask, resized_image_mask))).show()

    return full_image_mask, resized_image_mask


# Median absolute deviation
def MAD(data):
    median = np.median(data)
    sub = np.absolute(np.subtract(data, median))
    mad = np.median(sub) * 1.4826
    return mad


# funzione che identifica come outlier pixel che stanno al di fuori della distribuzione corrente descritta
# da una matrice di mean e una matrice di std per ogni pixel dell'immagine 256X256
def find_anomalies(data, mean, std, x, img_size):  # data è una image
    # define a list to accumlate anomalies
    anomalies_mask = np.zeros((img_size, img_size))
    # Set upper and lower limit to 3 standard deviation
    anomaly_cut_off = std * x

    var = std * std
    var = Image.fromarray(np.uint8(var))

    # limit = mediana - x * MAD
    lower_limit = mean - anomaly_cut_off
    upper_limit = mean + anomaly_cut_off
    # Generate outliers
    for i in range(img_size):
        for j in range(img_size):
            if data[i][j] > upper_limit[i][j] or data[i][j] < lower_limit[i][j]:
                anomalies_mask[i][j] = 255  # anomaly bianco
            else:
                anomalies_mask[i][j] = 0  # nero normale

    return anomalies_mask


# come funzione precedente soltanto che utilizza un unica distribuzione descritta da unico valore di mean e std

def find_anomalies_single_value(data, mean, std, x, img_size):  # data è una image
    # define a list to accumlate anomalies
    anomalies_mask = np.zeros((img_size, img_size))
    # Set upper and lower limit to 3 standard deviation
    anomaly_cut_off = std * x

    # limit = mediana - x * MAD
    lower_limit = mean - anomaly_cut_off
    upper_limit = mean + anomaly_cut_off
    # Generate outliers
    for i in range(img_size):
        for j in range(img_size):
            if data[i][j] > upper_limit or data[i][j] < lower_limit:
                anomalies_mask[i][j] = 255  # anomaly bianco
            else:
                anomalies_mask[i][j] = 0  # nero normale

    return anomalies_mask


# come funzione precedente soltanto che utilizza il percentile come threshold per determinare gli outliers
def find_anomalies_q(data, q1, q3, x, img_size, switch, mad):  # data è una image
    # define a matrix to accumlate anomalies
    anomalies_mask = np.zeros((img_size, img_size))
    # Set upper and lower limit to  [Q1 - 1.5 * IQR: Q3 + 1.5 * IQR]
    iqr = q3 - q1
    anomaly_cut_off = x * iqr

    # limit = mediana - x * MAD
    lower_limit = q1 - anomaly_cut_off
    upper_limit = q3 + anomaly_cut_off
    # Generate outliers
    for i in range(img_size):
        for j in range(img_size):
            if data[i][j] > upper_limit[i][j] or data[i][j] < lower_limit[i][j]:
                anomalies_mask[i][j] = 255  # anomaly bianco
            else:
                anomalies_mask[i][j] = 0  # nero normale

    return anomalies_mask


# come funzione precedente soltanto che utilizza il MAD
def find_anomalies_MAD(data, mean, std, x, img_size, switch, mad, mad_mask):  # data è una image
    # define a list to accumlate anomalies
    anomalies_mask = np.zeros((img_size, img_size))
    # Set upper and lower limit to 3 standard deviation
    anomaly_cut_off = std * x
    # limit = mediana - x * MAD
    lower_limit_mad = np.median(data) - x * mad
    upper_limit_mad = np.median(data) + x * mad
    lower_limit = mean - anomaly_cut_off
    upper_limit = mean + anomaly_cut_off
    # Generate outliers
    for i in range(img_size):
        for j in range(img_size):
            # standard
            if switch[i][j] == 1:
                if data[i][j] > upper_limit[i][j] or data[i][j] < lower_limit[i][j]:
                    anomalies_mask[i][j] = 255  # anomaly bianco
                else:
                    anomalies_mask[i][j] = 0  # nero normale
            # MAD
            else:
                if data[i][j] > upper_limit_mad[i][j] or data[i][j] < lower_limit_mad[i][j]:
                    anomalies_mask[i][j] = mad_mask[i][j]  # anomaly bianco
                else:
                    anomalies_mask[i][j] = 0  # nero normale

    return anomalies_mask


def get_seg_and_test_images(seg_images, test_images, path_seg, path_paz, img_width):
    transform = transforms.Compose([
        transforms.Resize((img_width, img_width)),
        transforms.ToTensor()])
    seg = sorted(glob.glob(path_seg + "/*.png"), key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    test = sorted(glob.glob(path_paz + "/*.png"), key=lambda x: int(x.split("\\")[-1].split(".")[0]))

    count_img = 0
    count_seg = 0
    for img in seg:
        image = Image.open(img)
        image = np.array(image)
        image[image > 0] = 255
        image = transform(Image.fromarray(image))
        image = torch.unsqueeze(image, dim=0)
        seg_images.append(image)

    for img in test:
        image = Image.open(img)
        image = transform(image)
        image = torch.unsqueeze(image, dim=0)

        test_images.append(image)

    return seg_images, test_images


def plot_heatmap(difference, path):
    matplotlib.pyplot.figure()
    sns.heatmap(difference, cmap="coolwarm")

    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.hist(np.array(difference).ravel(), bins=200, density=True)
    matplotlib.pyplot.xlabel("pixel values")
    matplotlib.pyplot.ylabel("relative frequency")
    matplotlib.pyplot.title("distribution of pixels")
    matplotlib.pyplot.imshow(difference)
    # matplotlib.pyplot.show()

    # matplotlib.pyplot.show()
    if save_heatmap: matplotlib.pyplot.savefig(path)


# mi serve solo per vedere se i pixel sono distribuiti normalmente
import gc


# funzione che serve per determinare data una matrice di media e di std per ogni pixel per la griglia 256x256
# quante di queste distribuzioni segue un andamento normale
def check_normal_distribution_in_residuals(model_path, patient_path, save_path, save_path_horizontally,
                                           test_images, seg_images, normal_pixels_shapiro, abnormal_pixels_shapiro,
                                           normal_pixels_agostino, abnormal_pixels_agostino):
    batch_size = 1
    img_width = 256
    img_height = img_width

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    # plant = 'pepper'
    brain = patient_path + '/brainmetshare/metshare/'
    plantVillageTrain, plantVillageTestHealthy, plantVillageTestDiseased, plantVillageTestDiseasedMaskSeg = generateDataset(
        brain, transform)

    trainloader = get_training_dataloader(plantVillageTrain, batch_size)

    num_hiddens = 256
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 256
    num_embeddings = 512

    commitment_cost = 0.30

    decay = 0.99
    # non utilizzato

    learning_rate = 1e-3

    # initialize the model
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    residual_volume = list()

    # mi serve per rimpiccolire binary map della std
    std_mask = np.zeros((img_width, img_width))

    # mi calcolo tutti i residual e creo le due matrici di std e mean
    for j, data in (enumerate(trainloader)):
        img = data['image']
        reconstruction, loss, perplexity = model(img)

        for x in range(trainloader.batch_size):
            rec = reconstruction[x, 0, :, :].cpu().detach().numpy()
            # rec[rec<0] = 0

            original = img[x, 0, :, :].cpu().detach().numpy()

            _, pa_bi = cv2.threshold(np.array(original), 0, 255, cv2.THRESH_BINARY)

            std_mask += pa_bi / 255

            diff = np.subtract(original, np.array(rec), dtype=np.float64)
            residual_volume.append(diff)

    pixels = list()
    std_mask[std_mask > 0] = 1

    for i in range(img_width):
        for j in range(img_width):
            for index in range(len(residual_volume)):

                pixels.append(residual_volume[index][i][j])
                if len(pixels) == len(
                        residual_volume):  # se entro vuol dire ho fatto una lista dello stesso pixel di tutte le imm

                    # faccio test normalità solo per i pixels che appartengono effettivamente agli slice di cervello
                    if std_mask[i][j] == 1:
                        # print('\nPixels riga: ',i,' Colonna: ',j)
                        # Shapiro Test
                        stat, p = shapiro(pixels)
                        alpha = 0.05
                        if p > alpha:
                            # print('Shapiro Sample looks Gaussian (fail to reject H0)')
                            normal_pixels_shapiro += 1
                        else:
                            # print('Shapiro Sample does not look Gaussian (reject H0)')
                            abnormal_pixels_shapiro += 1
                            plot_histogram(pixels)

                        # D'agostino K2 test
                        stat, p = normaltest(pixels)
                        # interpret
                        alpha = 0.05
                        if p > alpha:
                            # print('agostino Sample looks Gaussian (fail to reject H0)')
                            normal_pixels_agostino += 1
                        else:
                            # print('agostino Sample does not look Gaussian (reject H0)')
                            abnormal_pixels_agostino += 1
                    pixels.clear()
    return normal_pixels_shapiro, abnormal_pixels_shapiro, normal_pixels_agostino, abnormal_pixels_agostino


# su tutti i pazienti di test indica quante distribuzioni di residui in totale sono normali o non
def distribution_check_code(patients_name_list, patients_anomaly_mask_saving_path,
                            patients_anomaly_mask_horizontally_saving_path,
                            test_images, seg_images, patients_models_folders_path, patients_image_folders_path):
    normal_pixels_shapiro = 0
    abnormal_pixels_shapiro = 0
    normal_pixels_agostino = 0
    abnormal_pixels_agostino = 0
    for i in range(len(patients_name_list)):
        print('\nEvaluating patient: ', patients_name_list[i])

        normal_pixels_shapiro, abnormal_pixels_shapiro, normal_pixels_agostino, abnormal_pixels_agostino = check_normal_distribution_in_residuals(
            patients_models_folders_path[i], patients_image_folders_path[i],
            patients_anomaly_mask_saving_path[i],
            patients_anomaly_mask_horizontally_saving_path[i], test_images,
            seg_images, normal_pixels_shapiro, abnormal_pixels_shapiro, normal_pixels_agostino,
            abnormal_pixels_agostino)

        print('Total number of pixels examined: ', abnormal_pixels_agostino + normal_pixels_agostino)
        print('Total number of normal pixels shapiro :', normal_pixels_shapiro)
        print('Total number of abnormal pixels shapiro :', abnormal_pixels_shapiro)
        print('Total number of normal pixels agostino :', normal_pixels_agostino)
        print('Total number of abnormal pixels agostino :', abnormal_pixels_agostino)

    print('Total number of pixels examined: ', abnormal_pixels_agostino + normal_pixels_agostino)
    print('Total number of normal pixels shapiro :', normal_pixels_shapiro)
    print('Total number of abnormal pixels shapiro :', abnormal_pixels_shapiro)
    print('Total number of normal pixels agostino :', normal_pixels_agostino)
    print('Total number of abnormal pixels agostino :', abnormal_pixels_agostino)


# questa funzione calcola le varie distribuzioni di residui nel caso di matrici di mean e std o la singola distribuzione
# e ne calcola il dice score prendendo in input un modello di un autoencoder già alleanato
def compute_statistic(model_path, patient_path, save_path, save_path_horizontally, test_images, seg_images):
    batch_size = 1
    img_width = 196
    img_height = img_width

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    # plant = 'pepper'
    brain = patient_path + '/brainmetshare/metshare/'
    plantVillageTrain, plantVillageTestHealthy, plantVillageTestDiseased, plantVillageTestDiseasedMaskSeg = generateDataset(
        brain, transform)

    trainloader = get_training_dataloader(plantVillageTrain, batch_size)

    num_hiddens = 128
    num_residual_layers = 2
    num_residual_hiddens = 32
    num_embeddings = 64
    embedding_dim = 128
    commitment_cost = 0.30
    decay = 0
    # non utilizzato

    learning_rate = 1e-3

    # initialize the model
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    residual_volume = list()

    # mi serve per rimpiccolire binary map della std
    std_mask = np.zeros((img_width, img_width))

    # serve per vedere quanti pixel sono utilizzati per slice devi printarlo con show grey
    pixels_count = np.zeros((img_width, img_width))

    count_batch = 0
    # mi calcolo tutti i residual e creo le due matrici di std e mean
    for j, data in tqdm(enumerate(trainloader), total=int(len(plantVillageTrain) / trainloader.batch_size)):
        img = data['image']
        reconstruction, loss, perplexity = model(img)

        for x in range(trainloader.batch_size):
            # Image.fromarray(np.uint8(img[x, 0, :, :] * 255)).show()
            rec = reconstruction[x, 0, :, :].cpu().detach().numpy()
            rec[rec < 4 / 255] = 0
            # rec[rec<0] = 0

            original = img[x, 0, :, :].cpu().detach().numpy()
            c = np.copy(original)
            diff_mask = np.copy(original)

            c[c * 255 > 10] = 1
            pixels_count += c

            _, pa_bi = cv2.threshold(np.array(original), 0, 255, cv2.THRESH_BINARY)
            std_mask += pa_bi / 255

            diff = np.subtract(original, np.array(rec), dtype=np.float64)
            # la parte commentata serve per la single residual distribution
            '''diff = np.array(diff).flatten()
            diff  = diff[diff != 0]
            if count_batch == 0:
                flat_residual_distribution = np.copy(diff)

            else:flat_residual_distribution = np.concatenate((flat_residual_distribution,diff),axis=0)'''
            '''diff_mask[diff_mask>0]=1
            diff = diff * diff_mask'''

            residual_volume.append(diff)
        count_batch += 1

    '''single_mean = np.mean(flat_residual_distribution)
    single_std = np.std(flat_residual_distribution)'''
    std_mask[std_mask > 0] = 1
    if short_statistic: return std_mask

    pixels = list()
    std = np.zeros((img_width, img_width))
    mean = np.zeros((img_width, img_width))
    mad = np.zeros((img_width, img_width))
    # q1 = np.zeros((img_width, img_width))
    # q3 = np.zeros((img_width, img_width))
    # iqr = np.zeros((img_width, img_width))
    # median = np.zeros((img_width, img_width))

    # lo uso per tenere conto se utilizzare mad o standard gaussian
    # 0 mad 1 standard

    # rimetti a zero quando utilizzi mad e scommenta sotto
    switch = np.zeros((img_width, img_width))
    mad_mask = np.zeros((img_width, img_width))

    for i in tqdm(range(img_width)):
        for j in range(img_width):
            for index in range(len(residual_volume)):

                pixels.append(residual_volume[index][i][j])
                if len(pixels) == len(
                        residual_volume):  # se entro vuol dire ho fatto una lista dello stesso pixel di tutte le imm
                    # print('\nPixels riga: ',i,' Colonna: ',j)
                    # Shapiro Test
                    stat, p = shapiro(pixels)
                    alpha = 0.05
                    # eliminalo se usi mad

                    if MAD_on:
                        if p > alpha:
                            # print('Shapiro Sample looks Gaussian (fail to reject H0)')
                            switch[i][j] = 1
                        else:
                            # print('Shapiro Sample does not look Gaussian (reject H0)')
                            # ho MAD
                            mad[i][j] = MAD(pixels)
                            outliers_scores = 0
                            lower_limit_mad = np.median(pixels) - 3 * mad[i][j]
                            upper_limit_mad = np.median(pixels) + 3 * mad[i][j]

                            for z in range(len(pixels)):
                                if (pixels[z] - np.median(pixels)) / mad[i][j] > upper_limit_mad or \
                                        (pixels[z] - np.median(pixels)) / mad[i][j] < lower_limit_mad:
                                    outliers_scores += 1
                                    # Generate outliers

                            mad_mask[i][j] = 255 * (1 - (outliers_scores / len(pixels)))  # anomaly bianco

                    else:
                        switch[i][j] = 1

                    std[i][j] = np.std(pixels, dtype=np.float64)
                    mean[i][j] = np.mean(pixels)
                    # q3[i][j], q1[i][j] = np.percentile(pixels, [85, 15])
                    # iqr[i][j] = q3[i][j] - q1[i][j]
                    # median[i][j] = np.median(pixels)
                    pixels.clear()

    ##Image.fromarray(np.uint8(diff * 255)).show()
    if MAD_on: Image.fromarray(np.uint8(mad_mask)).save(
        r'C:\Users\paoli\Desktop\Tesi\Mad_mask' + '/' + patient_path.split('/')[-1] + '.png')

    # plot_heatmap(mean,save_path_horizontally + 'mean.png')
    # plot_heatmap(std,save_path_horizontally + 'std.png' )
    # used for lower and upper bound of statistic

    # for mean and std approach
    value = np.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5])
    value = np.array([3.0])

    # value = np.array([3.0])

    # for itq approach
    # value = np.array([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])

    if MAD_on: value = np.array([4])

    # value = np.array([3.0])

    k_collection = np.zeros(np.shape(value)[0])
    first_factor = np.zeros(np.shape(value)[0])
    second_factor = np.zeros(np.shape(value)[0])

    f_factor = []

    # Liste per salvarmi tutti i fattori dei parametri
    images = list()
    if len(value) > 1:
        # Create List of empty list
        for i in range(len(value)):
            f_factor.append([])

        s_factor = []

        # Create List of empty list
        for i in range(len(value)):
            s_factor.append([])

        # print('Lunghezza test set: ',len(testloaderDiseasedMask))

        for u in tqdm(range(len(test_images))):

            real_mask = seg_images[u][0, 0, :, :]

            reconstruction, loss, perplexity = model(test_images[u])
            # print(loss*1000)
            rec = np.array(reconstruction[0, 0, :, :].cpu().detach().numpy()) * 255

            test_image = np.array(test_images[u][0, 0, :, :]) * 255

            # print('Image number: ', u)

            diff = np.subtract(test_image, rec, dtype=np.float64)
            # diff = diff.clip(min=0)

            full_mask, resized_mask = create_images_masks(test_images[u][0, 0, :, :])
            # full_mask, resized_mask = create_images_masks(resized_mask)

            count = 0
            for x in value:
                maps = find_anomalies_single_value(diff / 255, single_mean, single_std, x, img_width)
                '''if MAD_on: maps = find_anomalies_MAD(diff / 255, mean, std, x, img_width, switch, mad, mad_mask)
                else: maps = find_anomalies(diff / 255, mean,std, x, img_width)'''

                # resized_mask,full_mask min,max 0,255
                mu = (resized_mask / 255) * (maps)
                # mu min max 0,255

                intersection, blue_pixels = check_std(std=std_mask, anomaly_map=mu / 255)
                # intersection,yellow min,max = 0,1

                mu = intersection * 255

                # mu = ((np.bitwise_and((resized_mask / 255).astype(int), (maps / 255).astype(int))) * 255).astype(int)
                # np.shape(real_mask) = 256,256 min max 0,1

                dice_loss, a, b = new_dice(real_mask, mu / 255)
                k_collection[count] += dice_loss
                first_factor[count] += a
                second_factor[count] += b
                f_factor[count].append(a)
                s_factor[count].append(b)
                count += 1
                '''images.append(Image.fromarray(np.uint8(test_image)))
                images.append(Image.fromarray(np.uint8(rec)))
                images.append(Image.fromarray(np.uint8(diff)))
                images.append(Image.fromarray(np.uint8(real_mask*255)))
                images.append(Image.fromarray(np.uint8(Image.fromarray(np.uint8(maps)))))'''

                # print('Evaluated image with k = ',x)
                # plot_images(images)
                # showImagesHorizontally(images)
                # images.clear()

        count = 0
        max_k = 0
        k = 0
        for x in value:
            if (k_collection[count] / len(test_images)) > max_k:
                max_k = (k_collection[count] / len(test_images))
                k = x

            print('For value of k: ', x,
                  ' the dice loss is: ', (k_collection[count] / len(test_images)),
                  ' with first factor: ', (first_factor[count] / len(test_images)),
                  ' with second factor: ', (second_factor[count] / len(test_images)), '\n')

            count += 1
    else:
        k = value
        max_k = 0

    print('For value of k: ', k, ' we have the maximum dice score of: ', max_k)

    anomaly_mask_images = list()
    anomaly_mask_images.clear()

    d = list()
    f = list()
    s = list()

    blue_pixels_list = []

    for i in tqdm(range(len(seg_images))):
        real_mask = seg_images[i][0, 0, :, :]

        reconstruction, loss, perplexity = model(test_images[i])

        rec = np.array(reconstruction[0, 0, :, :].cpu().detach().numpy()) * 255
        # rec[rec<0] = 0

        test_image = np.array(test_images[i][0, 0, :, :]) * 255

        diff = np.subtract(test_image, rec, dtype=np.float64)

        # plot_heatmap(diff, i)

        # diff = diff.clip(min=0)
        def optimal_k(diff, seg_images, single_mean, single_std):
            a = np.abs(np.array(seg_images) * diff)
            a = np.min(a[np.nonzero(a)])
            # metti tutto in abs anche k
            kappa = (a - single_mean) / single_std
            return kappa

        optimal_K = optimal_k(diff / 255, seg_images[i][0, 0, :, :], single_mean, single_std)
        full_mask, resized_mask = create_images_masks(test_images[i][0, 0, :, :].cpu())

        images.clear()
        maps = find_anomalies_single_value(diff / 255, single_mean, single_std, k, img_width)
        '''if MAD_on: maps = find_anomalies_MAD(diff / 255,mean,std, k, img_width,switch,mad,mad_mask)

        else: maps = find_anomalies(diff / 255, mean, std, k, img_width)'''

        # maps min,max 0,255

        mu = (resized_mask / 255) * (maps)
        intersection, blue_pixels = check_std(std=std_mask, anomaly_map=mu / 255)
        # Image.fromarray(np.uint8(blue_pixels * 255)).show()

        mu = intersection * 255

        # mu min,max 0,255

        # Image.fromarray(np.uint8(mu * 255)).show()
        dice_loss, a, b = new_dice(real_mask, mu / 255)
        d.append(dice_loss)
        f.append(a)
        s.append(b)

        '''images.append(Image.fromarray(np.uint8(test_image)))
        images.append(Image.fromarray(np.uint8(rec)))
        images.append(Image.fromarray(np.uint8(diff)))
        images.append(Image.fromarray(np.uint8(real_mask * 255)))
        images.append(Image.fromarray(np.uint8(Image.fromarray(np.uint8(maps)))))'''

        images.append(test_image)
        images.append(rec)
        images.append(diff)
        images.append(real_mask)
        images.append(mu / 255)
        images.append(blue_pixels)

        blue_pixels_list.append(blue_pixels)

        anomaly_mask_images.append((Image.fromarray(np.uint8(Image.fromarray(np.uint8(mu))))))

        # print('Evaluated image with k = ', k)
        # print('Dice : ', dice_loss, ' First factor: ', a, ' FSecond factor ', b)

        showImagesHorizontally(images, save_path_horizontally + str(i) + '.png')
    # print('The general scores of the entire testsets are: ')

    print('Dice: ', np.sum(d) / len(seg_images), ' First factor: ', np.sum(f) / len(seg_images), ' second factor: ',
          np.sum(s) / len(seg_images))

    for i in tqdm(range(len(anomaly_mask_images))):
        three_channels_AM(np.array(anomaly_mask_images[i]) / 255, np.array(seg_images[i][0, 0, :, :]),
                          blue_pixels_list[i]).save(save_path + str(i) + '.png')

        '''del residual_volume
    del pixels
    del images
    gc.collect()'''
    return std_mask


# partendo da più anomaly maps di più autoencoders ne restituisce una singola tramite la tecnica del consenso
def get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images, image_dim, number_of_models,
                              save, path_to_save_ensemble_images, seg_images, ensemble_mask, im_dim, test_images):
    '''ensemble_mask = np.zeros((im_dim,im_dim))
    for i in range(len(max_slice_all_patients)):
        ensemble_mask += max_slice_all_patients[i]


    ensemble_mask[ensemble_mask < len(max_slice_all_patients)] = 0
    ensemble_mask[ensemble_mask>=len(max_slice_all_patients)] = 1'''

    cv2.imwrite(r'C:\Users\paoli\Desktop\Tesi\Ensemble_mask.png', ensemble_mask * 255)

    # print(np.shape(anomaly_mask_from_all_models))
    ensemble_anomalies_map = []
    blue_pixels_list = []

    print("Calculating ensemble ...")

    for img in tqdm(range(number_of_images)):
        ensemble_image = np.zeros((image_dim, image_dim))  # creo immagine tutta nera

        for i in range(image_dim):

            for j in range(image_dim):
                count_whites = 0
                count_greys = 0
                for k in range(number_of_models):  # k is the k-esimo paziente
                    if anomaly_mask_from_all_models[k][img][i][j] == 255:
                        count_whites += 1
                    elif anomaly_mask_from_all_models[k][img][i][j] > 0 and anomaly_mask_from_all_models[k][img][i][
                        j] < 255:
                        count_greys += 1

                if count_whites + count_greys > number_of_models // 2:
                    if count_whites > count_greys:
                        ensemble_image[i][j] = 255
                    else:
                        ensemble_image[i][j] = 150

        # devi ricavarti blue pixels

        intersection, blue_pixels = check_std(std=ensemble_mask, anomaly_map=ensemble_image / 255)
        blue_pixels_list.append(blue_pixels)
        ensemble_anomalies_map.append(intersection * 255)
        # showImagesHorizontally(s)

    coloured_anomaly_maps = []
    print('\nConverting anomaly maps to to color images...')
    images = []
    for i in tqdm(range(len(ensemble_anomalies_map))):
        images.clear()
        # Image.fromarray((np.uint8(ensemble_anomalies_map[i]))) to visualize
        anomaly_map = three_channels_AM(ensemble_anomalies_map[i] / 255, seg_images[i][0, 0, :, :], blue_pixels_list[i])
        coloured_anomaly_maps.append(anomaly_map)

        # original
        images.append(np.array(test_images[i][0, 0, :, :]).copy())
        # seg
        images.append(np.array(seg_images[i][0, 0, :, :]))
        # anomaly maps
        images.append(anomaly_map.copy())
        # togliere i due commenti se vuoi salvare
        # scommenta se non stai ottimizzando ensemble
        # showImagesHorizontally_ensemble(images,path_to_save_ensemble_images +'_horizontal' + '/' + str(i) + '.png')

        # if save:
        # scommenta se non stai ottimizzando ensemble
        # coloured_anomaly_maps[i].save(path_to_save_ensemble_images + '/' + str(i) + '.png')

    return coloured_anomaly_maps


# partendo dalla anomaly maps in bianco e vero, ne viene generata una a colori con
# pixels verdi,rossi,blu e bianchi
def three_channels_AM(img, seg, blue_pixels):
    # blue_pixels indicano i pixel di cui il modello non sa nulla a riguardo
    # do imm in input 256*256 normalizzata
    # img = img[0,0,:,:]
    dim = np.shape(img)[0]
    stacked_img = np.stack((img,) * 3, axis=-1)

    for i in range(dim):
        for j in range(dim):
            if blue_pixels[i][j] == 1:
                stacked_img[i][j][0] = 0
                stacked_img[i][j][1] = 0
                stacked_img[i][j][2] = 1
            else:
                # img[i][j] > 0 corretto per mad sennò sarebbe stato == 1
                if seg[i][j] == 1 and img[i][j] > 0:  # anomaly detected like the mask said, put green
                    stacked_img[i][j][0] = 0
                    stacked_img[i][j][1] = 1
                    stacked_img[i][j][2] = 0

                if seg[i][j] == 1 and img[i][j] == 0:  # the model didn't detect the anomaly said by the mask put red
                    stacked_img[i][j][0] = 1
                    stacked_img[i][j][1] = 0
                    stacked_img[i][j][2] = 0

    # stacked_img = np.ascontiguousarray(stacked_img.transpose(1, 2, 0))
    stacked_img = stacked_img * 255
    stacked_img = Image.fromarray((np.uint8(stacked_img)), 'RGB')
    return stacked_img


def get_patients_images_path(path):
    # return a list that contains the paths of all the patients
    # list[0] contains patient 1 folder
    l = []
    for patient_id_folder in next(os.walk(path))[1]:
        l.append(str(path + '/' + patient_id_folder))
        '''for MRI_type in next(os.walk(os.path.join(path1, patient_id_folder)))[1]:
            if MRI_type == '2':
                path2 = next(os.walk(os.path.join(path1, patient_id_folder)))[0]
                for img in next(os.walk(os.path.join(path2, MRI_type)))[2]:'''
    l = sorted(l, key=lambda x: int(x.split("_")[-1]))
    return l


def get_patients_models_path(path, l):
    l = []
    for patient_id_folder in next(os.walk(path))[2]:
        l.append(str(path + '/' + patient_id_folder))

    l = sorted(l, key=lambda x: int(x.split("_")[-1]))
    return l


def get_choosed_patients_models_path(path, list):
    l = []
    for patient_id_folder in list:
        l.append(str(path + '/' + patient_id_folder))

    l = sorted(l, key=lambda x: int(x.split("_")[-1]))
    return l


# data una anomaly map in input questa funzione elimina tutti i pixel anomaly che non
# non hanno nessun pixel anomalo nel loro intorno di raggio 1 pixel
def eliminate_radius_in_ensemble_anomaly_map(list_anomaly_maps, path, test_images, seg_images):
    # blue_pixels indicano i pixel di cui il modello non sa nulla a riguardo
    # do imm in input 256*256 normalizzata
    # img = img[0,0,:,:]
    dice_list.clear()
    first_list.clear()
    second_list.clear()
    images = []
    # print('\nScores are after radius cleaning operation ')

    for u in range(len(list_anomaly_maps)):
        images.clear()
        # img = list_anomaly_maps[u]
        image_dim = np.shape(list_anomaly_maps[u])[0]
        # stacked_img = np.stack((img,) * 3, axis=-1)

        for i in range(image_dim):
            for j in range(image_dim):
                if list_anomaly_maps[u][i, j, 1] > 0:

                    count = 0
                    if 0 < i < image_dim and 0 < j < image_dim:
                        count += list_anomaly_maps[u][i - 1, j - 1, 1]
                        count += list_anomaly_maps[u][i - 1, j, 1]
                        count += list_anomaly_maps[u][i - 1, j + 1, 1]

                        count += list_anomaly_maps[u][i, j - 1, 1]
                        count += list_anomaly_maps[u][i, j + 1, 1]

                        count += list_anomaly_maps[u][i + 1, j - 1, 1]
                        count += list_anomaly_maps[u][i + 1, j, 1]
                        count += list_anomaly_maps[u][i + 1, j + 1, 1]

                    elif (i == 0) and (0 < j < image_dim):
                        count += list_anomaly_maps[u][i, j - 1, 1]
                        count += list_anomaly_maps[u][i, j + 1, 1]
                        count += list_anomaly_maps[u][i + 1, j - 1, 1]
                        count += list_anomaly_maps[u][i + 1, j, 1]
                        count += list_anomaly_maps[u][i + 1, j + 1, 1]

                    elif (i == image_dim) and (0 < j < image_dim):
                        count += list_anomaly_maps[u][i, j - 1, 1]
                        count += list_anomaly_maps[u][i, j + 1, 1]
                        count += list_anomaly_maps[u][i - 1, j - 1, 1]
                        count += list_anomaly_maps[u][i - 1, j, 1]
                        count += list_anomaly_maps[u][i - 1, j + 1, 1]

                    elif (j == 0) and (0 < i < image_dim):
                        count += list_anomaly_maps[u][i - 1, j, 1]
                        count += list_anomaly_maps[u][i + 1, j, 1]
                        count += list_anomaly_maps[u][i - 1, j + 1, 1]
                        count += list_anomaly_maps[u][i, j + 1, 1]
                        count += list_anomaly_maps[u][i + 1, j + 1, 1]

                    elif (j == image_dim) and (0 < i < image_dim):
                        count += list_anomaly_maps[u][i - 1, j, 1]
                        count += list_anomaly_maps[u][i + 1, j, 1]
                        count += list_anomaly_maps[u][i - 1, j - 1, 1]
                        count += list_anomaly_maps[u][i, j - 1, 1]
                        count += list_anomaly_maps[u][i + 1, j - 1, 1]

                    elif i == 0 and j == 0:
                        count += list_anomaly_maps[u][0, 1, 1]
                        count += list_anomaly_maps[u][1, 1, 1]
                        count += list_anomaly_maps[u][1, 0, 1]

                    elif i == image_dim and j == image_dim:
                        count += list_anomaly_maps[u][i, j - 1, 1]
                        count += list_anomaly_maps[u][i - 1, j - 1, 1]
                        count += list_anomaly_maps[u][i - 1, j, 1]

                    elif i == image_dim and j == 0:
                        count += list_anomaly_maps[u][i, j + 1, 1]
                        count += list_anomaly_maps[u][i - 1, j + 1, 1]
                        count += list_anomaly_maps[u][i - 1, j, 1]

                    elif i == 0 and j == image_dim:
                        count += list_anomaly_maps[u][i, j - 1, 1]
                        count += list_anomaly_maps[u][i + 1, j - 1, 1]
                        count += list_anomaly_maps[u][i + 1, j, 1]

                    if count == 0:
                        list_anomaly_maps[u][i, j, 0] = 0
                        list_anomaly_maps[u][i, j, 1] = 0
                        list_anomaly_maps[u][i, j, 2] = 0

        # original
        images.append(np.array(test_images[u][0, 0, :, :]).copy())
        # seg
        images.append(np.array(seg_images[u][0, 0, :, :]))
        # anomaly maps
        images.append(Image.fromarray(list_anomaly_maps[u], 'RGB').copy())

        # scommenta se non stai ottimizzando ensamble
        showImagesHorizontally_ensemble(images, path + '/' + str(u) + '.png')

    print('\nThe general scores of the entire testsets are after radius cleaning operation is: ')

    print('Dice: ', np.sum(dice_list) / len(list_anomaly_maps), ' First factor: ',
          np.sum(first_list) / len(list_anomaly_maps), ' second factor: ',
          np.sum(second_list) / len(list_anomaly_maps))

    count_missed_anomaly = 0
    for i in range(len(dice_list)):
        if dice_list[i] == 0: count_missed_anomaly += 1
    print('Missed anomalies: ', count_missed_anomaly, 'Spotted anomalies: ',
          len(list_anomaly_maps) - count_missed_anomaly, 'Total anomalies :', len(list_anomaly_maps))

    return np.sum(dice_list) / len(list_anomaly_maps), np.sum(first_list) / len(list_anomaly_maps), np.sum(
        second_list) / len(list_anomaly_maps)


# data in input un anomaly map, ogni pixel di quest'ultima viene ingrandito a 3x3 pixels anomali
def create_3X3_anomalies_map(list_anomaly_maps, seg_images):
    for i in range(len(seg_images)):
        seg_images[i] = np.squeeze(seg_images[i])

    for u in range(len(list_anomaly_maps)):
        # img = list_anomaly_maps[u]
        image_dim = np.shape(list_anomaly_maps[u])[0]
        # stacked_img = np.stack((img,) * 3, axis=-1)

        # Mi serve per tenere una copia delle posizioni originali delle anomaly maps
        original_mask = np.copy(np.array(list_anomaly_maps[u][:, :, 1]))
        # show_RGB(list_anomaly_maps[u])
        if original_mask.max() != 0:

            for i in range(image_dim):
                for j in range(image_dim):
                    if original_mask[i, j] == 255:

                        if 0 < i < image_dim and 0 < j < image_dim:
                            list_anomaly_maps[u][i - 1, j - 1, 1] = 255
                            list_anomaly_maps[u][i - 1, j, 1] = 255
                            list_anomaly_maps[u][i - 1, j + 1, 1] = 255

                            list_anomaly_maps[u][i, j - 1, 1] = 255
                            list_anomaly_maps[u][i, j + 1, 1] = 255

                            list_anomaly_maps[u][i + 1, j - 1, 1] = 255
                            list_anomaly_maps[u][i + 1, j, 1] = 255
                            list_anomaly_maps[u][i + 1, j + 1, 1] = 255

                        elif (i == 0) and (0 < j < image_dim):
                            list_anomaly_maps[u][i, j - 1, 1] = 255
                            list_anomaly_maps[u][i, j + 1, 1] = 255
                            list_anomaly_maps[u][i + 1, j - 1, 1] = 255
                            list_anomaly_maps[u][i + 1, j, 1] = 255
                            list_anomaly_maps[u][i + 1, j + 1, 1] = 255

                        elif (i == image_dim) and (0 < j < image_dim):
                            list_anomaly_maps[u][i, j - 1, 1] = 255
                            list_anomaly_maps[u][i, j + 1, 1] = 255
                            list_anomaly_maps[u][i - 1, j - 1, 1] = 255
                            list_anomaly_maps[u][i - 1, j, 1] = 255
                            list_anomaly_maps[u][i - 1, j + 1, 1] = 255

                        elif (j == 0) and (0 < i < image_dim):
                            list_anomaly_maps[u][i - 1, j, 1] = 255
                            list_anomaly_maps[u][i + 1, j, 1] = 255
                            list_anomaly_maps[u][i - 1, j + 1, 1] = 255
                            list_anomaly_maps[u][i, j + 1, 1] = 255
                            list_anomaly_maps[u][i + 1, j + 1, 1] = 255

                        elif (j == image_dim) and (0 < i < image_dim):
                            list_anomaly_maps[u][i - 1, j, 1] = 255
                            list_anomaly_maps[u][i + 1, j, 1] = 255
                            list_anomaly_maps[u][i - 1, j - 1, 1] = 255
                            list_anomaly_maps[u][i, j - 1, 1] = 255
                            list_anomaly_maps[u][i + 1, j - 1, 1] = 255

                        elif i == 0 and j == 0:
                            list_anomaly_maps[u][0, 1, 1] = 255
                            list_anomaly_maps[u][1, 1, 1] = 255
                            list_anomaly_maps[u][1, 0, 1] = 255

                        elif i == image_dim and j == image_dim:
                            list_anomaly_maps[u][i, j - 1, 1] = 255
                            list_anomaly_maps[u][i - 1, j - 1, 1] = 255
                            list_anomaly_maps[u][i - 1, j, 1] = 255

                        elif i == image_dim and j == 0:
                            list_anomaly_maps[u][i, j + 1, 1] = 255
                            list_anomaly_maps[u][i - 1, j + 1, 1] = 255
                            list_anomaly_maps[u][i - 1, j, 1] = 255

                        elif i == 0 and j == image_dim:
                            list_anomaly_maps[u][i, j - 1, 1] = 255
                            list_anomaly_maps[u][i + 1, j - 1, 1] = 255
                            list_anomaly_maps[u][i + 1, j, 1] = 255

        blue_pixels = np.zeros((image_dim, image_dim))
        for i in range(image_dim):
            for j in range(image_dim):
                if list_anomaly_maps[u][i, j, 0] == 0 and list_anomaly_maps[u][i, j, 1] == 1 and list_anomaly_maps[u][
                    i, j, 2] == 255:
                    blue_pixels[i, j] = 255
        # blue_pixels = np.copy(list_anomaly_maps[u][:,:,2])
        yas = three_channels_AM(list_anomaly_maps[u][:, :, 1] / 255, seg_images[u], blue_pixels / 255)

        yas.save(r'C:\Users\paoli\Desktop\Tesi\3x3 filtered image' + '/' + str(u) + '.png')
        # show_RGB(list_anomaly_maps[u])
        # show_RGB(original_mask)

        # show_RGB(yas)
        # print('ciao')


def get_patients_name_list(path):
    l = []

    for patient_id_folder in next(os.walk(path))[1]:
        l.append(str(patient_id_folder))
        '''for MRI_type in next(os.walk(os.path.join(path1, patient_id_folder)))[1]:
            if MRI_type == '2':
                path2 = next(os.walk(os.path.join(path1, patient_id_folder)))[0]
                for img in next(os.walk(os.path.join(path2, MRI_type)))[2]:'''

    l = sorted(l, key=lambda x: int(x.split("_")[-1]))
    return l


def create_saving_folders(path, patients_list):
    patients_anomaly_mask_saving_path = []
    for i in range(len(patients_list)):
        path_o = path + '/' + patients_list[i] + '/'
        if not os.path.isdir(path_o):
            os.mkdir(path_o)
        patients_anomaly_mask_saving_path.append(path_o)
    return patients_anomaly_mask_saving_path


def create_saving_folders_patients(path, patients_list):
    patients_anomaly_mask_saving_path = []
    for i in range(len(patients_list)):
        path_o = path + '/' + patients_list[i] + '/'
        if not os.path.isdir(path_o):
            os.mkdir(path_o)
            path_o1 = path_o + 'brainmetshare' + '/'
            os.mkdir(path_o1)
            path_o1 = path_o1 + 'metshare' + '/'
            os.mkdir(path_o1)
            os.mkdir(path_o1 + 'test' + '/')
            os.mkdir(path_o1 + 'test' + '/' + 'disease' + '/')
            os.mkdir(path_o1 + 'test' + '/' + 'disease' + '/' + 'id' + '/')
            os.mkdir(path_o1 + 'test' + '/' + 'disease' + '/' + 'id' + '/' + '2' + '/')
            os.mkdir(path_o1 + 'test' + '/' + 'disease' + '/' + 'id' + '/' + 'seg' + '/')

            os.mkdir(path_o1 + 'test' + '/' + 'healthy' + '/')
            os.mkdir(path_o1 + 'test' + '/' + 'healthy' + '/' + 'id' + '/')
            os.mkdir(path_o1 + 'test' + '/' + 'healthy' + '/' + 'id' + '/' + '2' + '/')

            os.mkdir(path_o1 + 'train' + '/')
            os.mkdir(path_o1 + 'train' + '/' + 'healthy' + '/')
            os.mkdir(path_o1 + 'train' + '/' + 'healthy' + '/' + 'id' + '/')
            os.mkdir(path_o1 + 'train' + '/' + 'healthy' + '/' + 'id' + '/' + '2' + '/')

        patients_anomaly_mask_saving_path.append(path_o)
    return patients_anomaly_mask_saving_path


def create_list_with_anomaly_mask_from_ensemble(ensemble_anomaly_mask_saving_path):
    l = []

    path = ensemble_anomaly_mask_saving_path
    # for k in range(len(ensemble_anomaly_mask_saving_path)):
    # path = ensemble_anomaly_mask_saving_path[k]

    # cv_img.clear()
    for img in sorted(glob.glob(path + "/*.png"), key=lambda x: int(x.split("\\")[-1].split(".")[0])):
        n = np.array(cv2.imread(img))
        n = n[..., ::-1]

        # Prendo tutti i valori RGB della anomaly map
        l.append(n)
        # l è la lista di tutte le immagini, mentre s è la lista dei
    return l


def create_list_with_anomaly_mask_from_all_models(number_of_models, patients_anomaly_mask_saving_path):
    l = []
    for i in range(number_of_models):
        l.append([])

    s = []

    for k in range(len(patients_anomaly_mask_saving_path)):
        path = patients_anomaly_mask_saving_path[k]
        s.append(path)
        # cv_img.clear()
        for img in sorted(glob.glob(path + "/*.png"), key=lambda x: int(x.split("\\")[-1].split(".")[0])):
            n = np.array(cv2.imread(img))
            n = n[..., ::-1]

            # I take only the green value because it has the same value of the original anomaly map grayscale
            l[k].append(n[:, :, 1])

        # l è la lista di tutte le immagini, mentre s è la lista dei
        # path da dove sono state prese le imm, giusto per capire l'ordine dei pazienti nella lista
    return l, s


def define_metrics(true_mask, pred_mask):
    pred_mask[pred_mask > 0] = 1
    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)

    a = pred_mask.sum()

    intersection = np.logical_and(true_mask, pred_mask)

    b = intersection.sum()

    # ^ è bitwise xor sarebbe il meno
    anomaly_pixel_outside_the_mask = pred_mask ^ intersection
    anomaly_pixel_outside_the_mask = anomaly_pixel_outside_the_mask.sum()
    pred_mask = pred_mask.sum()
    true_mask = true_mask.sum()
    intersection = intersection.sum()

    # pixel anomali accesi dentro la maschera dati dal modello/pixel della maschera
    dice_score = intersection / true_mask

    false_negative = true_mask - intersection

    true_positive = intersection

    false_positive = pred_mask - true_positive

    # What proportion of true anomalies was identified?(higher is better)
    recall = true_positive / (true_positive + false_negative)
    # Recall è anche uguale a true positive rate TPR

    # What proportion of identified anomalies are true anomalies?(higher is better)
    precision = true_positive / (true_positive + false_positive)

    # F1 Score identifies the overall performance of the anomaly detection model by combining(higher is better)
    # both Recall and Precision, using the harmonic mean
    f1_score = (2 * recall * precision) / (precision + recall)

    # FNR false negative rate, numero anomalie missate / numero totale di anomalie IDENTIFICATE DAL MODELLO!!!
    # se FNR è uguale a 0.6 questo vuol dire che il modello ha missato il 60% delle anomalie
    FNR = false_negative / pred_mask

    # FPR false positive rate, numero di false anomalie / numero totale di anomalie ESISTENTI!!!
    # se FPR è uguale a 0.4, questo vuol dire che il 40% delle anomalie detected dal modello non sono anomalie
    FPR = false_positive / true_mask

    # The ROC curve plots the true positive rate versus the false positive rate

    # The precision-recall curve, like the name implies, plots the precision versus the recall

    # precision-recall area-under-curve (PR-AUC)

    return recall, precision, f1_score, FNR, FPR


def center_image(img):
    '''cv2.imshow("Start", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    # get shape
    hh, ww = img.shape

    # get contours (presumably just one around the nonzero pixels)
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)

    # recenter
    startx = (ww - w) // 2
    starty = (hh - h) // 2
    result = np.zeros_like(img)
    result[starty:starty + h, startx:startx + w] = img[y:y + h, x:x + w]

    # view result
    '''cv2.imshow("RESULT", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return startx, starty, x, y, w, h
    # cv2.imwrite('white_shape_centered.png', result)


# funzione che centra le immagini del dataset
def centering_dataset():
    path_dataset = r'C:\Users\paoli\Desktop\Tesi\Dataset_splittato_completo_Met_brain'
    # test,train,val
    for type in next(os.walk(path_dataset))[1]:
        for elem in next(os.walk(os.path.join(path_dataset, type)))[1]:
            if str(elem) == 'healthy':
                path = path_dataset + '/' + type + '/' + elem
                for patient in next(os.walk(path))[1]:
                    print('ciao')
                    for risonance_type in next(os.walk(os.path.join(path, patient)))[1]:
                        if risonance_type == '2':
                            count = 0
                            for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:

                                n = np.array(cv2.imread(path + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                a = np.count_nonzero(n != 0)
                                if a > count:
                                    count = a
                                    biggest_image = n

                            startx, starty, x, y, w, h = center_image(biggest_image)

                            for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:
                                n = np.array(cv2.imread(path + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                # cv2.imshow("Start", n)
                                result = np.zeros_like(n)
                                result[starty:starty + h, startx:startx + w] = n[y:y + h, x:x + w]
                                # cv2.imwrite('path + '/' + patient + '/' + risonance_type + '/' + img', result)

            elif str(elem) == 'disease':
                path = path_dataset + '/' + type + '/' + elem
                for patient in next(os.walk(path))[1]:
                    print('ciao')
                    for risonance_type in next(os.walk(os.path.join(path, patient)))[1]:
                        if risonance_type == '2':
                            count = 0
                            for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:

                                n = np.array(cv2.imread(path + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                a = np.count_nonzero(n != 0)
                                if a > count:
                                    count = a
                                    biggest_image = n

                            startx, starty, x, y, w, h = center_image(biggest_image)

                            for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:
                                n = np.array(cv2.imread(path + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                cv2.imshow("Start", n)
                                result = np.zeros_like(n)
                                result[starty:starty + h, startx:startx + w] = n[y:y + h, x:x + w]
                                cv2.imshow("RESULT", result)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                # cv2.imwrite('path + '/' + patient + '/' + risonance_type + '/' + img', result)

                            for img in next(os.walk(os.path.join(path, patient, str('seg'))))[2]:
                                n = np.array(cv2.imread(path + '/' + patient + '/' + str('seg') + '/' + img, 0))
                                cv2.imshow("Start", n)
                                result = np.zeros_like(n)
                                result[starty:starty + h, startx:startx + w] = n[y:y + h, x:x + w]
                                cv2.imshow("RESULT", result)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                # cv2.imwrite('path + '/' + patient + '/' + risonance_type + '/' + img', result)


def centering_dataset():
    path_dataset = r'C:\Users\paoli\Desktop\Tesi\Dataset_splittato_completo_Met_brain'
    # test,train,val
    for type in next(os.walk(path_dataset))[1]:
        for elem in next(os.walk(os.path.join(path_dataset, type)))[1]:
            if str(elem) == 'healthy':
                path_healthy = path_dataset + '/' + type + '/' + elem
                path_disease = path_dataset + '/' + type + '/' + 'disease'

                for patient in next(os.walk(path_healthy))[1]:
                    for risonance_type in next(os.walk(os.path.join(path_healthy, patient)))[1]:
                        if risonance_type == '2':
                            count = 0
                            for img in next(os.walk(os.path.join(path_healthy, patient, risonance_type)))[2]:
                                n = np.array(
                                    cv2.imread(path_healthy + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                a = np.count_nonzero(n != 0)
                                if a > count:
                                    count = a
                                    biggest_image = n
                                    biggest_image_path = path_healthy + '/' + patient + '/' + risonance_type + '/' + img

                            for img in next(os.walk(os.path.join(path_disease, patient, risonance_type)))[2]:
                                n = np.array(
                                    cv2.imread(path_disease + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                a = np.count_nonzero(n != 0)
                                if a > count:
                                    count = a
                                    biggest_image = n
                                    biggest_image_path = path_disease + '/' + patient + '/' + risonance_type + '/' + img

                            show_grey(biggest_image)
                            startx, starty, x, y, w, h = center_image(biggest_image)
                            n = np.array(cv2.imread(biggest_image_path, 0))
                            result = np.zeros_like(n)
                            result[starty:starty + h, startx:startx + w] = n[y:y + h, x:x + w]
                            show_grey(result)
                            print('ciao')

                            for img in next(os.walk(os.path.join(path_healthy, patient, risonance_type)))[2]:
                                n = np.array(
                                    cv2.imread(path_healthy + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                # cv2.imshow("Start", n)
                                result = np.zeros_like(n)
                                result[starty:starty + h, startx:startx + w] = n[y:y + h, x:x + w]
                                # cv2.imwrite(path_healthy + '/' + patient + '/' + risonance_type + '/' + img, result)

                            for img in next(os.walk(os.path.join(path_disease, patient, risonance_type)))[2]:
                                n = np.array(
                                    cv2.imread(path_disease + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                # cv2.imshow("Start", n)
                                result = np.zeros_like(n)
                                result[starty:starty + h, startx:startx + w] = n[y:y + h, x:x + w]
                                # cv2.imshow("RESULT", result)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()
                                # cv2.imwrite(path_disease + '/' + patient + '/' + risonance_type + '/' + img, result)

                            for img in next(os.walk(os.path.join(path_disease, patient, str('seg'))))[2]:
                                n = np.array(cv2.imread(path_disease + '/' + patient + '/' + str('seg') + '/' + img, 0))
                                # cv2.imshow("Start", n)
                                result = np.zeros_like(n)
                                result[starty:starty + h, startx:startx + w] = n[y:y + h, x:x + w]
                                '''cv2.imshow("RESULT", result)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()'''
                                # cv2.imwrite(path_disease + '/' + patient + '/' + str('seg') + '/' + img, result)


def create_folder_patients():
    path_dataset = r'C:\Users\paoli\Desktop\Tesi\Dataset_splittato_completo_Met_brain'
    patients_name = next(os.walk(r'C:\Users\paoli\Desktop\Tesi\Dataset_splittato_completo_Met_brain\test\disease'))[1]
    save_path = r'C:\Users\paoli\Desktop\Tesi\Tutti i pazienti splittati'
    folders_patients_path = create_saving_folders_patients(save_path, patients_name)

    for type in next(os.walk(path_dataset))[1]:
        for elem in next(os.walk(os.path.join(path_dataset, type)))[1]:
            if str(elem) == 'healthy':
                print('healthy')
                path_healthy = path_dataset + '/' + type + '/' + elem

                path = path_dataset + '/' + type + '/' + elem
                for patient in next(os.walk(path))[1]:
                    print(patient)

                    for risonance_type in next(os.walk(os.path.join(path, patient)))[1]:
                        if risonance_type == '2':

                            # questo è per le foto healthy, quindi copio tutto nelle cartelle di training !!!!
                            for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:
                                # C:\Users\paoli\Desktop\Tesi\Tutti i pazienti splittati\Mets_005\brainmetshare\metshare\train\healthy\id\2
                                n = np.array(
                                    cv2.imread(path_healthy + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                cv2.imwrite(
                                    save_path + '/' + patient + '/brainmetshare/metshare/train/healthy/id/2/' + img, n)



            # questo è per le foto disease, quindi copio tutto nelle cartelle di test 2 e seg !!!!
            elif str(elem) == 'disease':
                print(elem)

                path_disease = path_dataset + '/' + type + '/' + elem
                path = path_dataset + '/' + type + '/' + elem

                for patient in next(os.walk(path))[1]:
                    print(patient)

                    for risonance_type in next(os.walk(os.path.join(path, patient)))[1]:

                        if risonance_type == '2':

                            # questo è per le foto disease, quindi copio tutto nelle cartelle di test 2 e seg !!!!
                            for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:
                                # C:\Users\paoli\Desktop\Tesi\Tutti i pazienti splittati\Mets_005\brainmetshare\metshare\train\healthy\id\2
                                n = np.array(
                                    cv2.imread(path_disease + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                cv2.imwrite(
                                    save_path + '/' + patient + '/brainmetshare/metshare/test/disease/id/2/' + img, n)

                        if risonance_type == str('seg'):

                            # questo è per le foto disease, quindi copio tutto nelle cartelle di test 2 e seg !!!!
                            for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:
                                # C:\Users\paoli\Desktop\Tesi\Tutti i pazienti splittati\Mets_005\brainmetshare\metshare\train\healthy\id\2
                                n = np.array(
                                    cv2.imread(path_disease + '/' + patient + '/' + risonance_type + '/' + img, 0))
                                cv2.imwrite(
                                    save_path + '/' + patient + '/brainmetshare/metshare/test/disease/id/seg/' + img, n)


def create_csv_all_patients():
    img_width = 256
    img_height = 256

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])

    patients_name = next(os.walk(r'C:\Users\paoli\Desktop\Tesi\Dataset_splittato_completo_Met_brain\test\disease'))[1]
    for patient in patients_name:
        path_dataset = r'C:\Users\paoli\Desktop\Tesi\Tutti i pazienti splittati' + '/' + str(
            patient) + '/brainmetshare/metshare'
        _, _, _, _ = generateDataset(path_dataset, transform)


def find_the_biggest_image_of_the_dataset():
    mask = np.zeros((256, 256))
    biggest_images = []
    path_dataset = r'C:\Users\paoli\Desktop\Tesi\Dataset_splittato_completo_Met_brain'
    for type in next(os.walk(path_dataset))[1]:
        for elem in next(os.walk(os.path.join(path_dataset, type)))[1]:
            path = path_dataset + '/' + type + '/' + elem
            for patient in tqdm(next(os.walk(path))[1]):
                for risonance_type in next(os.walk(os.path.join(path, patient)))[1]:
                    if risonance_type == '2' and elem != 'disease':
                        count = 0
                        for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:
                            n = np.array(
                                cv2.imread(path + '/' + patient + '/' + risonance_type + '/' + img, 0))

                            mask += n / 255
                            a = np.count_nonzero(n != 0)
                            if a > count:
                                count = a
                                biggest_image = n
                        biggest_images.append(biggest_image)

    mask[mask > 0] = 255
    show_grey(mask)
    return mask
    '''for i in range(len(biggest_images)):
        show_grey(biggest_images[i])
    return biggest_images'''


def omogenize_images_brightness():
    good_image = r'C:\Users\paoli\Desktop\Tesi\ok.png'
    good_image = np.array(cv2.imread(good_image, 0))
    threshold = np.max(good_image)
    save_path = r'C:\Users\paoli\Desktop\Tesi\Pazienti_normalizzati'
    for type in next(os.walk(save_path))[1]:
        print(type)
        for elem in next(os.walk(os.path.join(save_path, type, 'brainmetshare', 'metshare', 'test')))[1]:
            if str(elem) == 'disease':
                save_disease = save_path + '/' + type + '/brainmetshare/metshare/test/' + elem + '/id/2'
                for img in next(os.walk(save_disease))[2]:
                    n = np.array(cv2.imread(save_disease + '/' + img, 0))
                    n = n / (np.max(n))
                    n = n * threshold
                    cv2.imwrite(save_disease + '/' + img, n)

            elif str(elem) == 'healthy':
                save_healthy = save_path + '/' + type + '/brainmetshare/metshare/train/' + elem + '/id/2'
                for img in next(os.walk(save_healthy))[2]:
                    n = np.array(cv2.imread(save_healthy + '/' + img, 0))
                    n = n / (np.max(n))
                    n = n * threshold
                    cv2.imwrite(save_healthy + '/' + img, n)


def tilt_correction_01(img):
    img = numpy.uint8(img)
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = numpy.zeros(img.shape, numpy.uint8)

    # find the biggest contour (c) by the area
    c = max(contours, key=cv2.contourArea)

    (x, y), (MA, ma), angle = cv2.fitEllipse(c)

    cv2.ellipse(img, ((x, y), (MA, ma), angle), color=(0, 255, 0), thickness=2)

    rmajor = max(MA, ma) / 2
    if angle > 90:
        angle -= 90
    else:
        angle += 96
    xtop = x + math.cos(math.radians(angle)) * rmajor
    ytop = y + math.sin(math.radians(angle)) * rmajor
    xbot = x + math.cos(math.radians(angle + 180)) * rmajor
    ybot = y + math.sin(math.radians(angle + 180)) * rmajor
    cv2.line(img, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (0, 255, 0), 3)

    show_grey(img)

    M = cv2.getRotationMatrix2D((x, y), angle - 90, 1)  # transformation matrix

    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)

    show_grey(img)


def count_anomalies_in_patients():
    diz = {}
    save_path = r'C:\Users\paoli\Desktop\Tesi\Tutti i pazienti splittati omogenei'
    for type in next(os.walk(save_path))[1]:
        print(type)
        for elem in next(os.walk(os.path.join(save_path, type, 'brainmetshare', 'metshare', 'test')))[1]:
            if str(elem) == 'disease':
                save_disease = save_path + '/' + type + '/brainmetshare/metshare/test/' + elem + '/id/seg'
                count = len(next(os.walk(save_disease))[2])
                diz[str(type)] = [count + 1, 0]
    diz = sorted(diz.items(), key=lambda x: x[1])
    print(diz)
    return diz


# count_anomalies_in_patients()
def std_from_saved_arrays(path):
    a = np.zeros((196, 196))
    for array in next(os.walk(path))[2]:
        a += np.load(path + '/' + array)
    # a[a>0] = 1
    return a


def std_from_choosed_saved_arrays(path, l):
    a = np.zeros((196, 196))
    for patient in l:
        a += np.load(path + '/' + patient + '.npy')
    return a


# codice per valutare l'intero framework
def start():
    # ensemble_mask = std_from_saved_arrays(r'C:\Users\paoli\Desktop\Tesi\Arrays')

    image_dim = 196

    # contains the name of patients use for the ensemble
    patients_name_list = get_patients_name_list(r'C:\Users\paoli\Desktop\Tesi\Pazienti norm e croppati')

    patients_image_folders_path = get_patients_images_path(r'C:\Users\paoli\Desktop\Tesi\Pazienti norm e croppati')

    patients_models_folders_path = get_patients_models_path(r'C:\Users\paoli\Desktop\Tesi\Modelli_crop_norm',
                                                            patients_name_list)
    batch_size = 1
    img_width = 196
    img_height = img_width

    path_to_save_ensemble_images = r'C:\Users\paoli\Desktop\Tesi\ensemble_anomaly_masks'

    patients_anomaly_mask_saving_path = create_saving_folders(r'C:\Users\paoli\Desktop\Tesi\Anomaly_masks',
                                                              patients_name_list)

    save_filtered_ensemble_path = r'C:\Users\paoli\Desktop\Tesi\Filtered_ensemble'

    patients_anomaly_mask_horizontally_saving_path = create_saving_folders(
        r'C:\Users\paoli\Desktop\Tesi\Anomaly_maps_horizontal', patients_name_list)

    # create list containing image test of all pateints
    seg_images = []
    test_images = []
    # sono le massime dimensioni di slice di ogni paziente min,max 0,1
    # max_slice_all_patients = []

    # for i in range(len(patients_models_folders_path)):
    get_seg_and_test_images(seg_images, test_images, r'C:\Users\paoli\Desktop\Tesi\test\seg_paz_malati',
                            r'C:\Users\paoli\Desktop\Tesi\test\imm_paz_malati', image_dim)

    '''distribution_check_code(patients_name_list, patients_anomaly_mask_saving_path,
                            patients_anomaly_mask_horizontally_saving_path,
                            test_images, seg_images, patients_models_folders_path, patients_image_folders_path)'''

    ensemble_mask = np.zeros((196, 196))
    if short_ensemble:
        if save_statistic:
            print('\n', 'Computing statistic ... ')
            for i in tqdm(range(len(patients_name_list))):
                print('Evaluating patient: ', patients_name_list[i])
                mask = compute_statistic(patients_models_folders_path[i], patients_image_folders_path[i],
                                         patients_anomaly_mask_saving_path[i],
                                         patients_anomaly_mask_horizontally_saving_path[i], test_images, seg_images)
                np.save(r'C:\Users\paoli\Desktop\Tesi\Arrays' + '/' + str(patients_name_list[i]) + r'.npy',
                        np.array(mask))
                ensemble_mask += mask


        else:
            ensemble_mask = std_from_saved_arrays(r'C:\Users\paoli\Desktop\Tesi\Arrays')

        ensemble_mask[ensemble_mask < len(patients_name_list)] = 0
        ensemble_mask[ensemble_mask >= len(patients_name_list)] = 1

        number_of_models = len(patients_models_folders_path)
        number_of_images = len(test_images)

        # in gray scale
        anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
            number_of_models,
            patients_anomaly_mask_saving_path)

        # qui sono a colori ricorda
        coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images, image_dim,
                                                          number_of_models,
                                                          save_ensemble, path_to_save_ensemble_images, seg_images,
                                                          ensemble_mask, image_dim, test_images)
        d = []
        f = []
        s = []

        for i in range(len(coloured_anomaly_maps)):
            dice_loss, a, b = new_dice(np.array(seg_images[i][0, 0, :, :]),
                                       np.array(coloured_anomaly_maps[i])[:, :, 1] / 255)
            recall, precision, f1_score, FNR, FPR = define_metrics(np.array(seg_images[i][0, 0, :, :]),
                                                                   np.array(coloured_anomaly_maps[i])[:, :, 1] / 255)
            print('Dice : ' + str(dice_loss) + ' First factor: ' + str(a) + ' Second factor ' + str(b) +
                  'Recall: ' + str(recall) + 'Precision: ' + str(precision) + 'F1_score: ' + str(f1_score) +
                  'FNR: ' + str(FNR) + 'FPR: ' + str(FPR))

            # print('Evaluated image with k = ', 1)
            d.append(dice_loss)
            f.append(a)
            s.append(b)

        print('\nThe general scores of the entire testsets are: ')

        print('Dice: ', np.sum(d) / len(seg_images), ' First factor: ', np.sum(f) / len(seg_images), ' second factor: ',
              np.sum(s) / len(seg_images))

    ensemble_anomaly_maps = create_list_with_anomaly_mask_from_ensemble(path_to_save_ensemble_images)

    list_anomalies_maps = eliminate_radius_in_ensemble_anomaly_map(ensemble_anomaly_maps, save_filtered_ensemble_path,
                                                                   test_images, seg_images)

    # create_3X3_anomalies_map(list_anomalies_maps,seg_images)


# funzione per calcolare manualmente l'angolo rotazione per corregere le immagini di un certo paziente che non sono in asse
def get_angle_for_tilt_operation(img):
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            points.append((x, y))
            if len(points) >= 2:
                cv2.line(img, points[-1], points[-2], (0, 255, 0), 1)
            cv2.imshow("image", img)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    points = []
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    print(points)
    # cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), (0, 255, 0), 3)

    show_grey(img)
    angle = get_angle(points[0], points[1])
    print(angle)

    img = Image.fromarray(np.uint8(img))

    angle = angle - 90

    rotate_img = img.rotate(angle)
    # print(np.shape(rotate_img))
    rotate_img.show()
    return angle


# esegue tilting sui pazienti ossia ruota le immagini intorno al loro asse
def tilt_patient(angle, path_dataset):
    for type in next(os.walk(path_dataset))[1]:
        for elem in next(os.walk(os.path.join(path_dataset, type)))[1]:
            path = path_dataset + '/' + type + '/' + elem
            for patient in next(os.walk(path))[1]:
                for risonance_type in next(os.walk(os.path.join(path, patient)))[1]:
                    for img in next(os.walk(os.path.join(path, patient, risonance_type)))[2]:
                        n = np.array(cv2.imread(path + '/' + patient + '/' + risonance_type + '/' + img, 0))
                        n = Image.fromarray(np.uint8(n))
                        # n.show()
                        rotate_img = n.rotate(angle)
                        # print(np.shape(rotate_img))
                        # rotate_img.show()
                        cv2.imwrite(path + '/' + patient + '/' + risonance_type + '/' + img, np.array(rotate_img))


def crop_images_in_order_to_reduce_black_pixels():
    mask = find_the_biggest_image_of_the_dataset()
    a = np.where(mask != 0)
    x1, y1, x2, y2 = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
    min = (x1, y1)
    max = (x2, y2)
    print("min:", min)
    print("max:", max)
    x1 = x1 - 17
    x2 = x2 + 17
    y2 = y2 + 1
    mask = Image.fromarray(np.uint8(mask))
    img2 = mask.crop((x1, y1, x2, y2))
    img2.show()
    print(np.shape(np.array(img2)))

    save_path = r'C:\Users\paoli\Desktop\Tutti i pazienti splittati omogenei'
    for type in next(os.walk(save_path))[1]:
        print(type)
        for elem in next(os.walk(os.path.join(save_path, type, 'brainmetshare', 'metshare', 'test')))[1]:
            if str(elem) == 'disease':
                save_disease = save_path + '/' + type + '/brainmetshare/metshare/test/' + elem + '/id/2'
                for img in next(os.walk(save_disease))[2]:
                    n = np.array(cv2.imread(save_disease + '/' + img, 0))
                    n = Image.fromarray(np.uint8(n))
                    n = n.crop((x1, y1, x2, y2))
                    n = np.array(n)
                    cv2.imwrite(save_disease + '/' + img, n)
                save_disease = save_path + '/' + type + '/brainmetshare/metshare/test/' + elem + '/id/seg'
                for img in next(os.walk(save_disease))[2]:
                    n = np.array(cv2.imread(save_disease + '/' + img, 0))
                    n = Image.fromarray(np.uint8(n))
                    n = n.crop((x1, y1, x2, y2))
                    n = np.array(n)
                    cv2.imwrite(save_disease + '/' + img, n)

            elif str(elem) == 'healthy':
                save_healthy = save_path + '/' + type + '/brainmetshare/metshare/train/' + elem + '/id/2'
                for img in next(os.walk(save_healthy))[2]:
                    n = np.array(cv2.imread(save_healthy + '/' + img, 0))
                    n = Image.fromarray(np.uint8(n))
                    n = n.crop((x1, y1, x2, y2))
                    n = np.array(n)
                    cv2.imwrite(save_healthy + '/' + img, n)
    print('Done')


def delete_small_images_from_dataset_met_brain():
    save_path = r'C:\Users\paoli\Desktop\Tutti i pazienti splittati omogenei'
    count_tot_im = 0
    count_under_pixel = 0
    tot_pixels = 196 * 196
    for type in next(os.walk(save_path))[1]:
        print(type)
        for elem in next(os.walk(os.path.join(save_path, type, 'brainmetshare', 'metshare', 'test')))[1]:

            if str(elem) == 'healthy':
                save_healthy = save_path + '/' + type + '/brainmetshare/metshare/train/' + elem + '/id/2'
                for img in next(os.walk(save_healthy))[2]:
                    n = np.array(cv2.imread(save_healthy + '/' + img, 0))
                    n = np.array(n)

                    count_tot_im += 1
                    if np.count_nonzero(n) < 0.15 * tot_pixels:
                        count_under_pixel += 1
                        file_path = save_healthy + '/' + img
                        os.remove(file_path)

            elif str(elem) == 'disease':
                save_disease = save_path + '/' + type + '/brainmetshare/metshare/test/' + elem + '/id/2'
                for img in next(os.walk(save_disease))[2]:
                    n = np.array(cv2.imread(save_disease + '/' + img, 0))
                    n = np.array(n)
                    count_tot_im += 2
                    if np.count_nonzero(n) < 0.15 * tot_pixels:
                        count_under_pixel += 1
                        file_path = save_disease + '/' + img
                        os.remove(file_path)
                        file_path = save_path + '/' + type + '/brainmetshare/metshare/test/' + elem + '/id/seg/' + img
                        os.remove(file_path)

    print('Tot immagini: ', count_tot_im, ' Tot elem rimossi: ', count_under_pixel)
    print('Done')


# varie funzioni per creare delle immagini con patch nere per il training del CE-VQ-VAE

def mask_random_square(img_shape, square_size, n_val, channel_wise_n_val=False, square_pos=None, x1=0, y1=0, x2=0,
                       y2=0):
    """Masks (sets = 0) a random square in an image"""

    img_h = img_shape[-2]
    img_w = img_shape[-1]

    img = np.zeros(img_shape)

    if square_pos is None:
        w_start = x1
        h_start = y1
    else:
        pos_wh = square_pos[np.random.randint(0, len(square_pos))]
        w_start = x1
        h_start = y1

    if img.ndim == 2:
        rnd_n_val = int(random.uniform(n_val))
        img[random.randint(h_start, y2 - square_size): (h_start + square_size),
        random.randint(w_start, x2 - square_size): (w_start + square_size)] = rnd_n_val
    elif img.ndim == 3:
        if channel_wise_n_val:
            for i in range(img.shape[0]):
                rnd_n_val = int(random.uniform(n_val[0], n_val[1]))
                img[i, h_start: (h_start + square_size), w_start: (w_start + square_size)] = rnd_n_val
        else:
            try:
                # code to run when exception occurs
                rnd_n_val = random.uniform(n_val[0], n_val[1])
                inity = random.randint(h_start, y2 - square_size)
                initx = random.randint(w_start, x2 - square_size)
                img[:, inity: (inity + square_size), initx: (initx + square_size)] = rnd_n_val
            # code that may cause exception
            except:
                print('error')
                return img

    elif img.ndim == 4:
        if channel_wise_n_val:
            for i in range(img.shape[0]):
                rnd_n_val = get_range_val(n_val)
                img[:, i, h_start: (h_start + square_size), w_start: (w_start + square_size)] = rnd_n_val
        else:
            rnd_n_val = get_range_val(n_val)
            img[:, :, h_start: (h_start + square_size), w_start: (w_start + square_size)] = rnd_n_val

    return img


def mask_random_squares(img_shape, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None, x1=0, y1=0,
                        x2=0, y2=0):
    """Masks a given number of squares in an image"""
    img = np.zeros(img_shape)
    for i in range(n_squares):
        img = mask_random_square(
            img_shape, square_size, n_val, channel_wise_n_val=channel_wise_n_val, square_pos=square_pos, x1=x1, y1=y1,
            x2=x2, y2=y2
        )
    return img


def random_black_square_images_CE():
    # 96 è la metà di 192 che è l'imm in input al modello
    def create_mask(noise_val, x1, y1, x2, y2):
        rnd_square_size = int(random.uniform(0, 96))
        rnd_n_squares = int(random.uniform(0, 3))

        new_img = mask_random_squares(
            [1, 196, 196],
            square_size=rnd_square_size,
            n_squares=rnd_n_squares,
            n_val=noise_val,
            channel_wise_n_val=False,
            square_pos=None,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )
        return new_img

    # tensor è l'imm

    ##################################################

    save_path = r'C:\Users\paoli\Desktop\Dataset_Finale_finale'
    '''for type in next(os.walk(save_path))[1]:

        for elem in next(os.walk(os.path.join(save_path, type)))[1]:
            if str(elem) == 'healthy':
                save_healthy = save_path + '/' + type + '/' + elem
                for patient in next(os.walk(save_healthy))[1]:
                    if int(patient) >= -1:
                        print(patient)
                        for img in next(os.walk(os.path.join(save_healthy,patient,str(2))))[2]:
                            n = np.array(cv2.imread(save_healthy + '/' + patient + '/2/' + img, 0)) / 255
                            a = np.where(n != 0)
                            x1, y1, x2, y2 = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
                            noise_val = (np.min(n[np.nonzero(n)] * 2), np.max(n)/2 )
                            ce_tensor = create_mask(noise_val,x1,y1,x2,y2)
                            inpt_noisy = np.where(ce_tensor[0] != 0, ce_tensor[0], n)
                            #show_grey(inpt_noisy * 255)
                            n = np.array(inpt_noisy * 255)
                            cv2.imwrite(save_healthy + '/' + patient + '/2/' + img, n)'''
    for patient_id_folder in sorted(next(os.walk(save_path))[1], key=lambda x: int(x.split("_")[-1])):
        print(patient_id_folder)

        path1 = save_path + '/' + patient_id_folder + '/brainmetshare/metshare/train/healthy/id/2'
        for img in sorted(next(os.walk(path1))[2], key=lambda x: int(x.split(".")[0])):
            n = np.array(
                cv2.imread(save_path + '/' + patient_id_folder + '/brainmetshare/metshare/train/healthy/id/2/' + img,
                           0)) / 255
            a = np.where(n != 0)

            x1, y1, x2, y2 = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
            noise_val = (np.min(n[np.nonzero(n)] * 2), np.max(n) / 2)
            ce_tensor = create_mask(noise_val, x1, y1, x2, y2)
            inpt_noisy = np.where(ce_tensor[0] != 0, ce_tensor[0], n)
            n = np.array(inpt_noisy * 255)
            cv2.imwrite(save_path + '/' + patient_id_folder + '/brainmetshare/metshare/train/healthy/id/2/' + img, n)

    print('Done')


def normalise_mask(array):
    narray = array[:] - np.min(array) / (np.max(array) - np.min(array))
    return narray


def create_black_patch(original_image, img_width):
    maschera = np.zeros((img_width, img_width))
    # show_grey(original_image)
    count = np.nonzero(original_image)
    min_x = np.min(count[0])
    min_y = np.min(count[1])
    max_x = np.max(count[0])
    max_y = np.max(count[1])
    maschera[min_x:max_x, min_y:max_y] = 255
    # show_grey(maschera)
    maschera = np.zeros((img_width, img_width))

    thirty_per_cent_x = int(((max_x - min_x) / 100) * 30)
    thirty_per_cent_y = int(((max_y - min_y) / 100) * 30)
    '''print('thirty_per_cent_x',thirty_per_cent_x)
    print('thirty_per_cent_y',thirty_per_cent_y)'''

    min_x_rand = np.random.randint(low=min_x, high=max_x - thirty_per_cent_x)
    min_y_rand = np.random.randint(low=min_y, high=max_y - thirty_per_cent_y)

    maschera[min_x_rand:min_x_rand + thirty_per_cent_x,
    min_y_rand:min_y_rand + thirty_per_cent_y] = 1

    # show_grey(maschera)

    maschera = np.logical_not(maschera) * 1
    patch_image = original_image * maschera
    patch_image = patch_image
    patch_image = torch.from_numpy(patch_image)
    patch_image = torch.unsqueeze(patch_image, dim=0)
    patch_image = torch.unsqueeze(patch_image, dim=0)
    # show_grey(original_image)

    return patch_image


def cevae_test():
    patients_image_folders_path = get_patients_images_path(r'C:\Users\paoli\Desktop\Tesi\Pazienti norm e croppati')

    batch_size = 1
    img_width = 256
    img_height = img_width

    # create list containing image test of all pateints
    seg_images = []
    test_images = []
    # sono le massime dimensioni di slice di ogni paziente min,max 0,1
    # max_slice_all_patients = []

    get_seg_and_test_images(seg_images, test_images, r'C:\Users\paoli\Desktop\Tesi\test\seg_paz_malati',
                            r'C:\Users\paoli\Desktop\Tesi\test\imm_paz_malati', img_width)

    # for i in range(len(seg_images)):
    #   Image.fromarray(np.uint8(np.squeeze(seg_images[i])*255)).save(r'C:\Users\paoli\Desktop\Tesi\seg_paz_malati' + '/' + str(i)+'.png')

    # for i in range(len(seg_images)):
    #   Image.fromarray(np.uint8(np.squeeze(test_images[i])*255)).save(r'C:\Users\paoli\Desktop\Tesi\imm_paz_malati' + '/' + str(i)+'.png')

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    # plant = 'pepper'
    # brain = patient_path + '/brainmetshare/metshare/'
    # plantVillageTrain, plantVillageTestHealthy, plantVillageTestDiseased, plantVillageTestDiseasedMaskSeg = generateDataset(
    #    brain, transform)

    # trainloader = get_training_dataloader(plantVillageTrain, batch_size)

    num_hiddens = 256  # larghezza latent space 512/1024
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 256  # provo prima questi 3   aumenta prima dimensione
    num_embeddings = 512

    commitment_cost = 0.25  # più alto + vicino al valore del codebook
    decay = 0

    # non utilizzato

    learning_rate = 1e-3

    # initialize the model
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay)
    model.load_state_dict(torch.load(r'C:\Users\paoli\Downloads\ce_vae_model', map_location=torch.device('cpu')))
    model.eval()

    masked_data = []
    for i in range(len(seg_images)):
        masked_data.append([])

    for i in range(len(seg_images)):

        original_image = np.array(np.squeeze(test_images[i]) * 255)
        original_image[original_image < 10] = 0

        for j in range(5):
            masked_data[i].append(create_black_patch(np.squeeze(original_image), img_width))

        # show_grey(maschera)
    c = list(zip(test_images, masked_data))
    random.shuffle(c)
    test_images, masked_data = zip(*c)

    for i in range(len(test_images)):
        show_grey(test_images[i])
        for u in range(5):
            show_grey(masked_data[i][u])

    ''''show_grey(test_images[0])
    show_grey(masked_data[0][0])

    show_grey(test_images[0])
    show_grey(masked_data[0][1])

    show_grey(test_images[2])
    show_grey(masked_data[0][2])

    show_grey(test_images[3])
    show_grey(masked_data[0][3])

    show_grey(test_images[4])
    show_grey(masked_data[0][4])'''

    '''
    for i in range(len(seg_images)):
       reconstruction, loss, perplexity = model(test_images[i])
       rec = np.squeeze(reconstruction.detach().numpy()*255)
       rec[rec<0] = 0
       #show_grey(rec)
       img = np.squeeze(test_images[i].detach().numpy())* 255
       #show_grey(img)
       full_mask, resized_mask = create_images_masks(test_images[i][0, 0, :, :].cpu())
       resized_mask = resized_mask/np.max(resized_mask)




       mask = rec - img
       mask = normalise_mask(mask)
       mask = mask * resized_mask*255
       mask[mask<50] = 0'''
    # plot_heatmap(mask,r'C:\Users\paoli\Desktop'+ '\i')


def release_list(a):
    del a[:]
    del a


def Resize(image, output_size):
    image = image[:, :, 0]
    import skimage.transform as skt
    resized = skt.resize(image, output_size, order=3, mode='reflect')
    return resized


def square_mask(image, margin, patchsize, input_size):
    """Generate mask tensor from bbox.
    Args:
    bbox: configuration tuple, (top, left, height, width)
    config: Config should have configuration including IMG_SHAPES,
    MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
    Returns:
    image shape inputted with just mask
    No ------tf.Tensor: output with shape [1, H, W, 1]
    """
    bboxs = []
    # for i in range(times):
    bbox = random_bbox(image, margin, patchsize)
    bboxs.append(bbox)
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros((height, width), np.float32)
    for bbox in bboxs:
        h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
        w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
        mask[(bbox[0] + h): (bbox[0] + bbox[2] - h), (bbox[1] + w): (bbox[1] + bbox[3] - w)] = 1.
        mask = np.expand_dims(mask, axis=2)
        mask = Resize(mask, input_size)
        mask = np.expand_dims(mask, axis=2)
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.from_numpy(mask)
    return mask.float()  # mask.reshape((1, ) + mask.shape).astype(np.float32)


def calculate_dice_anomalies_maps(list_anomaly_maps, seg_images):
    dice = []
    first = []
    second = []
    count_missed_anomalies = 0
    for u in range(len(list_anomaly_maps)):
        dice_loss, a, b = new_dice(np.squeeze(seg_images[u]), list_anomaly_maps[u] / 255)

        if a == 0: count_missed_anomalies += 1
        dice.append(dice_loss)
        first.append(a)
        second.append(b)

    print('Dice: ', np.sum(dice) / len(dice), ' First factor: ', np.sum(first) / len(dice),
          ' second factor: ', np.sum(second) / len(dice), 'Missed anomalies: ', count_missed_anomalies)


# funzione per calcolare la migliore combinazioni di ensemble di autoencoders

def iterative_combinations_ensemble(best_3, patients_name_list, number_of_models, image_dim, FALSE, seg_images,
                                    test_images):
    save_ensemble = False
    # questa funzione la chiami più volte
    best_dice = 0
    best_first = 0
    best_second = 0
    for patient in patients_name_list:
        best_3_copy = best_3.copy()

        if patient not in best_3:
            best_3_copy.append(patient)
            # print(best_3 , selected_patients)
            dice_list.clear()
            first_list.clear()
            second_list.clear()
            print(best_3_copy)

            patients_anomaly_mask_saving_path = create_saving_folders(r'C:\Users\paoli\Desktop\Tesi\Anomaly_masks',
                                                                      best_3_copy)

            # da sistemare questo in kaggle TO DO
            ensemble_mask = std_from_choosed_saved_arrays(r'C:\Users\paoli\Desktop\Tesi\Arrays', best_3_copy)

            path_to_save_ensemble_images = r'C:\Users\paoli\Desktop\Tesi\ensemble_anomaly_masks'

            save_filtered_ensemble_path = r'C:\Users\paoli\Desktop\Tesi\Filtered_ensemble'

            ensemble_mask[ensemble_mask < len(best_3_copy)] = 0
            ensemble_mask[ensemble_mask >= len(best_3_copy)] = 1

            number_of_models = len(best_3_copy)
            number_of_images = len(test_images)

            # in gray scale
            anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
                number_of_models,
                patients_anomaly_mask_saving_path)

            # qui sono a colori ricorda
            coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images,
                                                              image_dim,
                                                              number_of_models,
                                                              save_ensemble, path_to_save_ensemble_images,
                                                              seg_images,
                                                              ensemble_mask, image_dim, test_images)

            for m in range(len(coloured_anomaly_maps)):
                coloured_anomaly_maps[m] = np.array(coloured_anomaly_maps[m])
            # coloured_anomaly_maps[np.array(coloured_anomaly_maps)] = coloured_anomaly_maps

            dice, first, second = eliminate_radius_in_ensemble_anomaly_map(coloured_anomaly_maps,
                                                                           save_filtered_ensemble_path,
                                                                           test_images, seg_images)
            if dice > best_dice:
                best_dice = dice
                best_first = first
                best_second = second
                best_4 = best_3_copy.copy()

            print('Dice: ', dice, ' First factor: ', first, ' second factor: ', second)

    return best_4, best_dice, best_first, best_second


def choose_best_combination_of_ensemble():
    from copy import deepcopy

    image_dim = 196

    # contains the name of patients use for the ensemble
    patients_name_list = get_patients_name_list(r'C:\Users\paoli\Desktop\Tesi\Pazienti norm e croppati')

    patients_image_folders_path = get_patients_images_path(r'C:\Users\paoli\Desktop\Tesi\Pazienti norm e croppati')

    patients_models_folders_path = get_choosed_patients_models_path(r'C:\Users\paoli\Desktop\Tesi\Modelli_crop_norm',
                                                                    patients_name_list)

    # create list containing image test of all pateints
    seg_images = []
    test_images = []
    # sono le massime dimensioni di slice di ogni paziente min,max 0,1
    # max_slice_all_patients = []

    get_seg_and_test_images(seg_images, test_images, r'C:\Users\paoli\Desktop\Tesi\test\seg_paz_malati',
                            r'C:\Users\paoli\Desktop\Tesi\test\imm_paz_malati', image_dim)

    list_combinations = list(itertools.combinations(patients_name_list, 3))
    # codice per creare combinazione di 3 pazienti
    # np.save(r'C:\Users\paoli\Desktop\3_combinations' + r'.npy', list_combinations)

    count_best_patients_index = 0
    max_dice = 0
    # list_combinations = np.load(r'C:\Users\paoli\Desktop\3_combinations' + r'.npy', allow_pickle=True)
    for h in range(len(list_combinations)):
        selected_patients = list_combinations[h]
        dice_list.clear()
        first_list.clear()
        second_list.clear()

        print(selected_patients)

        patients_anomaly_mask_saving_path = create_saving_folders(r'C:\Users\paoli\Desktop\Tesi\Anomaly_masks',
                                                                  selected_patients)

        ensemble_mask = std_from_choosed_saved_arrays(r'C:\Users\paoli\Desktop\Tesi\Arrays', selected_patients)

        path_to_save_ensemble_images = r'C:\Users\paoli\Desktop\Tesi\ensemble_anomaly_masks'

        save_filtered_ensemble_path = r'C:\Users\paoli\Desktop\Tesi\Filtered_ensemble'

        ensemble_mask[ensemble_mask < len(selected_patients)] = 0
        ensemble_mask[ensemble_mask >= len(selected_patients)] = 1

        number_of_models = len(selected_patients)
        number_of_images = len(test_images)

        # in gray scale
        anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
            number_of_models,
            patients_anomaly_mask_saving_path)

        # qui sono a colori ricorda
        coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images, image_dim,
                                                          number_of_models,
                                                          save_ensemble, path_to_save_ensemble_images, seg_images,
                                                          ensemble_mask, image_dim, test_images)

        for m in range(len(coloured_anomaly_maps)):
            coloured_anomaly_maps[m] = np.array(coloured_anomaly_maps[m])
        # coloured_anomaly_maps[np.array(coloured_anomaly_maps)] = coloured_anomaly_maps

        dice, first, second = eliminate_radius_in_ensemble_anomaly_map(coloured_anomaly_maps,
                                                                       save_filtered_ensemble_path,
                                                                       test_images, seg_images)

        print('Dice: ', dice, ' First factor: ', first, ' second factor: ', second)
        if dice > max_dice:
            max_dice = dice
            count_best_patients_index = h

    best_combination = list(list_combinations[count_best_patients_index]).copy()

    for i in range(len(patients_name_list) - 3):
        # Prima di passare a kaggle devi implementare una funzione qui dentro
        best_combination, best_d, best_f, best_s = iterative_combinations_ensemble(best_combination.copy(),
                                                                                   patients_name_list, number_of_models,
                                                                                   image_dim, False, seg_images,
                                                                                   test_images)
        print(best_combination)
        print('Dice: ', best_d, ' First factor: ', best_f, ' second factor: ', best_s)


def create_folder_with_test_and_seg_images(path=r'C:\Users\paoli\Desktop\Tesi\Pazienti norm e croppati'):
    count_img = 0
    count_seg = 0
    for patient in next(os.walk(path))[1]:
        path_0 = path + '/' + patient + '/brainmetshare\metshare/test\disease\id/2'
        for img in next(os.walk(path_0))[2]:
            count_img += 1
            image = Image.open(path_0 + '/' + img)
            image.save(r'C:\Users\paoli\Desktop\Tesi\test\imm_paz_malati' + '/' + str(count_img) + '.png')

        path_0 = path + '/' + patient + '/brainmetshare\metshare/test\disease\id/seg'
        for img in next(os.walk(path_0))[2]:
            count_seg += 1
            image = Image.open(path_0 + '/' + img)
            image.save(r'C:\Users\paoli\Desktop\Tesi\test\seg_paz_malati' + '/' + str(count_seg) + '.png')


# create_folder_with_test_and_seg_images()
# start()

def clean_single_anomaly_single_pixels(list_anomaly_maps, image_dim):
    for u in range(len(list_anomaly_maps)):
        for i in range(image_dim):
            for j in range(image_dim):
                if list_anomaly_maps[u][i, j] > 0:

                    count = 0
                    if 0 < i < image_dim and 0 < j < image_dim:
                        count += list_anomaly_maps[u][i - 1, j - 1]
                        count += list_anomaly_maps[u][i - 1, j]
                        count += list_anomaly_maps[u][i - 1, j + 1]

                        count += list_anomaly_maps[u][i, j - 1]
                        count += list_anomaly_maps[u][i, j + 1]

                        count += list_anomaly_maps[u][i + 1, j - 1]
                        count += list_anomaly_maps[u][i + 1, j]
                        count += list_anomaly_maps[u][i + 1, j + 1]

                    elif (i == 0) and (0 < j < image_dim):
                        count += list_anomaly_maps[u][i, j - 1]
                        count += list_anomaly_maps[u][i, j + 1]
                        count += list_anomaly_maps[u][i + 1, j - 1]
                        count += list_anomaly_maps[u][i + 1, j]
                        count += list_anomaly_maps[u][i + 1, j + 1]

                    elif (i == image_dim) and (0 < j < image_dim):
                        count += list_anomaly_maps[u][i, j - 1]
                        count += list_anomaly_maps[u][i, j + 1]
                        count += list_anomaly_maps[u][i - 1, j - 1]
                        count += list_anomaly_maps[u][i - 1, j]
                        count += list_anomaly_maps[u][i - 1, j + 1]

                    elif (j == 0) and (0 < i < image_dim):
                        count += list_anomaly_maps[u][i - 1, j]
                        count += list_anomaly_maps[u][i + 1, j]
                        count += list_anomaly_maps[u][i - 1, j + 1]
                        count += list_anomaly_maps[u][i, j + 1]
                        count += list_anomaly_maps[u][i + 1, j + 1]

                    elif (j == image_dim) and (0 < i < image_dim):
                        count += list_anomaly_maps[u][i - 1, j]
                        count += list_anomaly_maps[u][i + 1, j]
                        count += list_anomaly_maps[u][i - 1, j - 1]
                        count += list_anomaly_maps[u][i, j - 1]
                        count += list_anomaly_maps[u][i + 1, j - 1]

                    elif i == 0 and j == 0:
                        count += list_anomaly_maps[u][0, 1]
                        count += list_anomaly_maps[u][1, 1]
                        count += list_anomaly_maps[u][1, 0]

                    elif i == image_dim and j == image_dim:
                        count += list_anomaly_maps[u][i, j - 1]
                        count += list_anomaly_maps[u][i - 1, j - 1]
                        count += list_anomaly_maps[u][i - 1, j]

                    elif i == image_dim and j == 0:
                        count += list_anomaly_maps[u][i, j + 1]
                        count += list_anomaly_maps[u][i - 1, j + 1]
                        count += list_anomaly_maps[u][i - 1, j]

                    elif i == 0 and j == image_dim:
                        count += list_anomaly_maps[u][i, j - 1]
                        count += list_anomaly_maps[u][i + 1, j - 1]
                        count += list_anomaly_maps[u][i + 1, j]

                    if count == 0:
                        list_anomaly_maps[u][i, j] = 0

    new_list_anomaly_maps = []
    for i in range(len(list_anomaly_maps)):
        new_list_anomaly_maps.append(Image.fromarray(np.uint8(Image.fromarray(np.uint8(list_anomaly_maps[i])))))

    return new_list_anomaly_maps


def calculate_dice_anomalies_maps(list_anomaly_maps, seg_images):
    dice = []
    first = []
    second = []
    for u in range(len(list_anomaly_maps)):
        dice_loss, a, b = new_dice(seg_images[u], list_anomaly_maps[u] / 255)
        dice.append(dice_loss)
        first.append(a)
        second.append(b)

    print('Dice: ', np.sum(dice) / len(dice), ' First factor: ', np.sum(first) / len(dice),
          ' second factor: ', np.sum(second) / len(dice))

    optimal_k = np.array([[0.13501051, 0.09004752, 0.042219274, 0.084915936, 0.18647186, 0.14020078, 0.05483786,
                           0.13688241, 0.049314775, 0.11209182, 0.102297485, 0.11927171, 0.1390257, 0.10616786,
                           0.07208812, 0.0766946, 0.03238158, 0.019293552, 0.07842468, 0.110179484, 0.121586084,
                           0.12013381, 0.1304351, 0.1356587, 0.13380146, 0.13331385, 0.122727156, 0.13352893,
                           0.09084341, 0.03170291, 0.03309957, 0.105585895, 0.13595526, 0.13303898, 0.06818903,
                           0.13030499, 0.117700465, 0.1202059, 0.050623465, 0.09719988, 0.1187507, 0.117830575,
                           0.028833603, 0.029555624, 0.10707802, 0.017191328, 0.047789834, 0.11598974, 0.11300198,
                           0.078245334, 0.050345734, 1.5646182, 0.05294428, 0.11913046, 0.10858772, 0.12075973,
                           0.06379764, 0.5946416, 0.087787144, 0.20430408, 0.020330887, 0.051068943, 0.0887524,
                           0.54211605, 1.5654106, 0.05690733, 0.034478523, 0.12724513, 0.1828951, 0.57770896,
                           0.11070459, 0.046421368, 0.01413917, 0.09956407, 0.35858655, 0.13197938, 0.11044848,
                           0.052714545, 0.010819032],
                          [0.123689115, 0.17212436, 1.2814422, 0.21807113, 0.27299473, 0.3270647, 0.18511698,
                           0.11700229, 0.16836211, 0.10849875, 0.28087765, 0.1723673, 0.30661398, 0.12517065,
                           0.12437279, 0.18917327, 0.1817457, 0.13334364, 0.2168518, 0.1423105, 0.1591324, 0.11968658,
                           0.11732556, 0.111543454, 0.136697, 0.12466021, 0.16171513, 0.2155674, 0.121831216,
                           0.38811225, 0.12082296, 0.12908821, 0.11026504, 0.121798694, 0.13294339, 0.11094408,
                           0.14232112, 0.114122875, 0.11961224, 0.10859433, 0.25297877, 0.13321686, 0.13493004,
                           0.1296305, 0.16114096, 0.23016502, 0.13192582, 0.11762624, 0.14185448, 0.13995278,
                           0.23798421, 0.75143105, 0.12125574, 0.16784702, 0.19290498, 0.18475989, 0.26629266,
                           0.14458124, 0.72493607, 0.14406615, 0.29536045, 1.2743459, 0.39146292, 1.018938, 0.3616604,
                           0.2709397, 0.18815638, 0.52396524, 0.1291599, 0.19574326, 0.21573399, 0.2581263, 0.19631144,
                           0.10862487, 0.28433788, 0.17510335, 0.14287403, 0.22832571, 0.16231252],
                          [0.027118608, 0.055707477, 1.1839484, 0.03727162, 0.13513136, 0.27223867, 0.2735272,
                           0.059335887, 0.015011481, 0.05529207, 0.07592223, 0.22597747, 0.115173124, 0.18386237,
                           0.032324035, 0.14458081, 0.03513029, 0.023767361, 0.14544694, 0.3894697, 0.01224387,
                           0.014749642, 0.016272029, 0.04184344, 0.025979318, 0.013766255, 0.035386126, 0.03963948,
                           0.032759763, 0.3371629, 0.024321685, 0.024550876, 0.073721595, 0.07501812, 0.07660113,
                           0.05108769, 0.054648135, 0.045698375, 0.11781215, 0.01577034, 0.021508101, 0.06408959,
                           0.1598673, 0.08804868, 0.03705509, 0.015659077, 0.011648907, 0.015819645, 0.023717394,
                           0.012996068, 0.04077544, 1.5012857, 0.07024109, 0.103670955, 0.32723176, 0.15406823,
                           0.15848015, 0.054261044, 0.5098587, 0.22108984, 0.5798062, 2.1095922, 0.07350507, 0.22743456,
                           0.32080975, 0.050560016, 0.12443536, 0.87892497, 0.6538864, 0.96727526, 0.0631182,
                           0.19496079, 0.17910999, 0.14480068, 0.24157444, 0.076138094, 0.0724797, 0.08518113,
                           0.23457544],
                          [0.09603036, 0.2172104, 0.17185588, 0.2388321, 0.240494, 0.37021372, 0.11574881, 0.13464543,
                           0.14228675, 0.55873376, 0.2486522, 0.13617605, 0.11767385, 0.14458679, 0.19599135,
                           0.124125406, 0.1687152, 0.1419135, 0.12786894, 0.11027921, 0.10700016, 0.09974916,
                           0.11178039, 0.112024695, 0.104623, 0.11559222, 0.102889284, 0.11724823, 0.09907098,
                           0.31920707, 0.13493979, 0.15049791, 0.1506763, 0.18217932, 0.17971031, 0.100713156,
                           0.1380746, 0.117995284, 0.21581459, 0.12948728, 0.10903706, 0.15229109, 0.15236174,
                           0.122912675, 0.10793854, 0.1367047, 0.20928122, 0.100207165, 0.14163682, 0.105842784,
                           0.12552357, 0.5532147, 0.14123063, 0.098424, 0.40592945, 0.1312457, 0.24559742, 0.13869569,
                           0.21766253, 0.11575528, 0.15964922, 1.5503769, 0.111078076, 0.33450502, 1.0777509,
                           0.17953075, 0.19175626, 0.6796189, 0.15268493, 0.68858945, 0.18021423, 0.19654001, 0.5147739,
                           0.11177215, 0.2588873, 0.172152, 0.17960788, 0.17132372, 0.1798251],
                          [0.11350512, 0.1084306, 0.12063309, 0.14500417, 0.11336506, 0.37959293, 0.14889109, 0.1551295,
                           0.11102542, 0.1488975, 0.23760515, 0.18264888, 0.121029586, 0.26052135, 0.18335491,
                           0.15083137, 0.21538666, 0.12061262, 0.4051688, 0.11049942, 0.13456737, 0.10891919,
                           0.115780495, 0.109528005, 0.11323269, 0.11009461, 0.114530884, 0.15666434, 0.15255293,
                           0.1513142, 0.12308625, 0.11553523, 0.11240324, 0.12069576, 0.1510795, 0.14646672, 0.11534243,
                           0.11922169, 0.13330369, 0.16876641, 0.11575747, 0.1170416, 0.11229005, 0.13447464,
                           0.13541088, 0.110954754, 0.29907036, 0.10828224, 0.11441002, 0.16835842, 0.16297182,
                           0.756825, 0.11132183, 0.10927731, 0.11609257, 0.20451885, 0.11134485, 0.2501581, 0.64445287,
                           0.15039074, 0.13003324, 0.92906535, 0.11707102, 0.41863182, 0.54562944, 0.50538445,
                           0.33093265, 0.31795314, 0.34540153, 0.33190212, 0.21150933, 0.21928383, 0.25450933,
                           0.21407825, 0.11483274, 0.37008595, 0.12772845, 0.1234706, 0.38685074],
                          [0.004783878, 0.0042843446, 0.15539835, 0.010687829, 0.13690397, 0.09661854, 0.010906418,
                           0.10392791, 0.033299427, 0.07874396, 0.023795616, 0.010898881, 0.65128505, 0.21391574,
                           0.0038039975, 0.014845812, 0.031270117, 0.02355236, 0.041728113, 0.1918458, 0.019019555,
                           0.01911206, 0.02050582, 0.009821012, 0.07145687, 0.008446438, 0.006377041, 0.07012135,
                           0.050659418, 0.024488384, 0.008861688, 0.0047167256, 0.015659867, 0.022179155, 0.0156242335,
                           0.016190922, 0.01799993, 0.12307463, 0.01613199, 0.024435623, 0.016288223, 0.087152086,
                           0.037671544, 0.19166766, 0.031856675, 0.0037354743, 0.115216404, 0.0924777, 0.01872559,
                           0.038747355, 0.081870325, 0.23675723, 0.012895645, 0.13661686, 0.28280476, 0.041366994,
                           0.1315811, 0.19753048, 0.121898785, 0.11465417, 0.33918217, 0.601577, 0.31994224, 0.59158224,
                           0.45155385, 0.29707676, 0.01830486, 0.074304685, 0.38550034, 1.0626389, 0.07962962,
                           0.06719336, 0.22996658, 0.05131039, 0.067373574, 0.03575221, 0.1575952, 0.07533801,
                           0.17922522],
                          [0.4032634, 0.35372308, 0.6008334, 0.56423527, 0.50082445, 0.30568305, 0.20249595, 0.35522896,
                           0.47016612, 0.4853698, 0.49145317, 0.4336614, 0.5794185, 0.55391365, 0.513633, 0.52107733,
                           0.57080233, 0.53675497, 0.49174893, 0.28967926, 0.56957495, 0.5878231, 0.5873199, 0.5720005,
                           0.52908564, 0.58912706, 0.5686889, 0.546837, 0.41917485, 0.24023128, 0.4588483, 0.55325705,
                           0.5861288, 0.5343645, 0.57463115, 0.59006774, 0.58346766, 0.5684768, 0.5810737, 0.5741618,
                           0.55751866, 0.58996516, 0.56807053, 0.5634223, 0.44242364, 0.5671582, 0.38575286, 0.58616436,
                           0.58203983, 0.22923632, 0.086905345, 1.4241602, 0.5821246, 0.49101305, 0.28970382,
                           0.49686706, 0.09367753, 0.34395036, 0.21045342, 0.048953447, 0.41751003, 2.2201302,
                           0.28869328, 0.22583649, 0.19651939, 0.5548494, 0.58841455, 0.80891085, 0.7288515, 0.28551847,
                           0.3218724, 0.29302767, 0.44146866, 0.5402559, 0.12810025, 0.54805034, 0.5571744, 0.5890499,
                           0.4397509],
                          [0.045862023, 0.024596479, 0.049060658, 0.06991088, 0.21237703, 0.094850615, 0.13213077,
                           0.010636477, 0.2199224, 0.0023112793, 0.19522767, 0.008504246, 0.41159034, 0.010040426,
                           0.019846603, 0.13887095, 0.041661277, 0.14202325, 0.10613274, 0.012608256, 0.00020702407,
                           0.020312322, 0.0038030276, 0.040742584, 0.0026595818, 0.002156211, 0.011622946, 0.0021760236,
                           0.030944515, 0.038779493, 0.011149696, 0.026734505, 0.0027597928, 0.005364115, 0.016590621,
                           0.101894915, 0.06352171, 0.0004300365, 0.019700052, 0.017849337, 8.190532e-05, 0.030565683,
                           0.05182891, 0.029178951, 0.00034141072, 0.0004069845, 0.27053636, 0.051256604, 0.038825832,
                           0.00062872015, 0.011247588, 0.44100133, 0.046643436, 0.09459458, 0.10397675, 0.09858506,
                           0.44690278, 0.8041247, 0.00075106084, 0.5258764, 0.29727122, 1.6680223, 0.19436458,
                           0.27962485, 0.35962203, 0.36307383, 0.0016784451, 0.54438645, 0.98660564, 0.17328787,
                           0.12488316, 0.00077816774, 0.007956852, 0.22031164, 0.5635968, 0.2312856, 0.1611838,
                           0.020062085, 0.12107862],
                          [0.28342453, 0.23109311, 0.30691916, 0.25000197, 0.111689925, 0.2864973, 0.19110638,
                           0.21265905, 0.26468894, 0.2693083, 0.00033041366, 0.27874324, 0.0077348393, 0.22311652,
                           0.30097693, 0.23549332, 0.14583024, 0.20260969, 0.2188402, 0.28370315, 0.30254722,
                           0.28114676, 0.30808884, 0.30767065, 0.2933733, 0.3031893, 0.29673964, 0.29198158,
                           0.108443275, 0.16293049, 0.22081327, 0.18199405, 0.23721202, 0.2743839, 0.29415873,
                           0.21390264, 0.30338112, 0.21260244, 0.28101408, 0.2706522, 0.20153496, 0.2679053, 0.28866428,
                           0.26865488, 0.20613009, 0.17393196, 0.13474013, 0.29233062, 0.2951209, 0.1967461,
                           0.060730264, 1.7798711, 0.31672645, 0.17383069, 0.2561341, 0.29285824, 0.80031264,
                           0.61429787, 0.669402, 0.4314422, 0.13422771, 1.6368377, 0.00037075864, 0.2534596, 0.19012608,
                           0.3500849, 0.05183364, 1.2765385, 0.77298564, 1.1728868, 0.0071976897, 0.25487322, 0.2635472,
                           0.29850358, 0.46433297, 0.16411276, 0.2810015, 0.2441321, 0.2827919],
                          [0.12297818, 0.13677666, 0.14946051, 0.114706345, 0.15567687, 0.45914754, 0.11324717,
                           0.1464517, 0.16505587, 0.31088033, 0.057890825, 0.15199561, 0.1465506, 0.08025807,
                           0.14049837, 0.11784607, 0.13501754, 0.06799087, 0.16271864, 0.1290654, 0.15767759, 0.166435,
                           0.16447774, 0.15937896, 0.13796565, 0.15168972, 0.13648981, 0.16671708, 0.053885832,
                           0.24442707, 0.16447386, 0.09864413, 0.14485624, 0.05707733, 0.13069655, 0.16360591,
                           0.13228904, 0.08441481, 0.14853334, 0.122825235, 0.16177659, 0.1552734, 0.12297996,
                           0.08786458, 0.15113868, 0.15213783, 0.11510626, 0.12749317, 0.13620476, 0.1531733,
                           0.0035807237, 0.7357174, 0.07551635, 0.1137661, 0.25178128, 0.08539195, 0.07868821,
                           0.16502711, 1.3083229, 0.022360068, 0.09334421, 1.9019363, 0.08024022, 0.04009807,
                           0.24629329, 0.06488141, 0.050101105, 0.06832054, 0.09118411, 0.6484051, 0.15884279,
                           0.073150344, 0.016055629, 0.14437601, 0.07735877, 0.11889642, 0.11634108, 0.0040800218,
                           0.12786807],
                          [0.07786601, 0.056787003, 0.3979114, 0.028065076, 0.32942355, 0.18485487, 0.0636806,
                           0.04464537, 0.040441856, 0.1055545, 0.30563644, 0.28486326, 1.0621837, 0.052620914,
                           0.21915837, 0.15458347, 0.03484133, 0.14545113, 0.078693874, 0.025348661, 0.03247971,
                           0.04314118, 0.037620574, 0.033438772, 0.034545667, 0.029336432, 0.09584517, 0.060848683,
                           0.18579082, 0.3792642, 0.025648946, 0.04268475, 0.023979368, 0.06626674, 0.034620505,
                           0.03040406, 0.14000815, 0.07022309, 0.0741776, 0.026226416, 0.045664486, 0.05288239,
                           0.08514489, 0.038394842, 0.05688864, 0.063425586, 0.18222992, 0.024573468, 0.048451122,
                           0.027333308, 0.08192216, 0.88532853, 0.29572338, 0.10952702, 0.09722925, 0.025915967,
                           0.07309473, 0.08906244, 0.3819326, 0.14327894, 0.19404078, 0.6360984, 0.3862225, 0.33379015,
                           0.66403687, 0.56461793, 0.09617686, 0.14868219, 0.12546887, 0.55482775, 0.104616225,
                           0.03523401, 0.10823141, 0.16198616, 0.32146558, 0.034110487, 0.27413437, 0.023729902,
                           0.29647273]])

    best_patients_for_every_images = []
    for i in range(np.shape(optimal_k)[1]):
        best_patients_for_every_images.append(np.argmax(optimal_k[:, i]))
        print('For image: ', i, ' best k is: ', optimal_k[best_patients_for_every_images[i], i],
              ' best index patient is: ', best_patients_for_every_images[i])


class Engine:
    def __init__(self,
                 model,
                 trainloader,
                 trainset,
                 testloader,
                 testset,
                 testloadermask,
                 testsetmask,
                 testloaderdiseased,
                 testdiseasedset,
                 epochs,
                 optimizer,
                 criterion,
                 scheduler,
                 device,
                 early_stopping,
                 img_width,
                 img_height,
                 compute_loss,
                 patience_dice,
                 training_set_per_shuffle,
                 masked_data):

        self.model = model
        self.trainloader = trainloader
        self.trainset = trainset
        self.testloader = testloader
        self.testset = testset
        self.testloadermask = testloadermask
        self.testsetmask = testsetmask
        self.testloaderdiseased = testloaderdiseased
        self.testdiseasedset = testdiseasedset
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.early_stopping = early_stopping
        self.img_width = img_width
        self.img_height = img_height
        self.compute_loss = compute_loss
        self.patience_dice = patience_dice
        self.training_set_per_shuffle = training_set_per_shuffle
        self.masked_data = masked_data

    # calculate mean and std of residuals
    def check_anomaly_score(self, model):
        std_mask = np.zeros((img_width, img_width))

        self.model.eval()
        img_width = self.img_width
        img_height = self.img_height

        residual_volume = list()

        # mi calcolo tutti i residual e creo le due matrici di std e mean

        for j, data in tqdm(enumerate(self.trainloader), total=int(len(self.trainset) / self.trainloader.batch_size)):
            img = data['image']
            img = img.cuda()
            reconstruction, loss, perplexity = model(img)

            for x in range(self.trainloader.batch_size):
                rec = reconstruction[x, 0, :, :].cpu().detach().numpy()
                rec[rec < 0] = 0
                # rec = rec.clip(min=0)
                original = img[x, 0, :, :].cpu()
                _, pa_bi = cv2.threshold(np.array(original), 0, 255, cv2.THRESH_BINARY)
                std_mask += pa_bi / 255
                diff = np.subtract(np.array(original), np.array(rec), dtype=np.float64)
                # diff = diff.clip(min=0)
                residual_volume.append(diff)

        pixels = list()
        std = np.zeros((img_height, img_width))
        mean = np.zeros((img_height, img_width))
        std_mask[std_mask > 0] = 1

        for i in tqdm(range(img_height)):
            for j in range(img_width):
                for index in range(len(residual_volume)):
                    pixels.append(residual_volume[index][i][j])
                    if len(pixels) == len(
                            residual_volume):  # se entro vuol dire ho fatto una lista dello stesso pixel di tutte le imm
                        numpy_pixels = np.array(pixels, dtype=np.float64)
                        std_0 = np.std(numpy_pixels, dtype=np.float64)
                        mean_0 = np.mean(numpy_pixels, dtype=np.float64)
                        std[i][j] = std_0
                        mean[i][j] = mean_0
                        pixels.clear()

        # used for lower and upper bound of statistic
        a = 0

        value = np.array([3])
        k_collection = np.zeros(np.shape(value)[0])
        first_factor = np.zeros(np.shape(value)[0])
        second_factor = np.zeros(np.shape(value)[0])

        f_factor = []

        # Liste per salvarmi tutti i fattori dei parametri

        # Create List of empty list
        for i in range(len(value)):
            f_factor.append([])

        s_factor = []

        # Create List of empty list
        for i in range(len(value)):
            s_factor.append([])
        images = list()

        total_dice_score = 0
        print(len(self.testloadermask))
        # devi aggiungere anche testloader diseased!
        # for i in tqdm(range(len(seg_images))):
        for item1, item2 in zip(enumerate(self.testloaderdiseased), enumerate(self.testloadermask)):

            i, data = item1
            u, data_seg = item2

            img = data['image'].cuda()
            seg = data_seg['image'].cuda()

            real_mask = seg[0, :, :].cpu().detach().numpy()
            # display(Image.fromarray(np.uint8(real_mask*255)))
            reconstruction, loss, perplexity = model(img)

            rec = np.array(reconstruction[0, 0, :, :].cpu().detach().numpy()) * 255
            rec[rec < 0] = 0

            test_image = np.array(img[0, 0, :, :].cpu().detach().numpy()) * 255

            print('Image number: ', i)

            '''print('Original Image')
            display(Image.fromarray(np.uint8(test_image)))'''

            '''print('Reconstructed Image')
            display(Image.fromarray(np.uint8(rec)))'''

            diff = np.subtract(test_image, rec, dtype=np.float64)
            # diff = diff.clip(min=0)

            '''print('Residual Image')
            display(Image.fromarray(np.uint8(diff)))'''

            full_mask, resized_mask = create_images_masks(img[0, 0, :, :].cpu())

            '''print('Evaluated image mask')
            display(Image.fromarray(np.uint8(resized_mask)))'''

            count = 0
            for x in value:
                # np.shape(maps) = 256,256 min max 0,255
                maps = find_anomalies(diff / 255, mean, std, x, img_width, img_height)

                mu = (resized_mask / 255) * (maps)
                # np.shape(mu) = 256,256 min max 0,255

                intersection, blue_pixels = check_std(std=std_mask, anomaly_map=mu / 255)
                mu = intersection * 255

                dice_loss, a, b = new_dice(real_mask, mu / 255)
                k_collection[count] += dice_loss
                first_factor[count] += a
                second_factor[count] += b
                f_factor[count].append(a)
                s_factor[count].append(b)
                count += 1
                # print('Evaluated image with k = ',x)
                # plot_images(images)
                # showImagesHorizontally(images)
                # images.clear()

        count = 0
        count_max = 0
        max_k = 0
        k = 0
        max_first = 0
        max_second = 0

        for x in value:

            if (k_collection[count] / len(self.testloaderdiseased)) > max_k:
                max_k = (k_collection[count] / len(self.testloaderdiseased))
                max_first = (first_factor[count] / len(self.testloaderdiseased))
                max_second = (second_factor[count] / len(self.testloaderdiseased))
                count_max = count
                k = x

            print('For value of k: ', x,
                  ' the dice loss is: ', (k_collection[count] / len(self.testloaderdiseased)),
                  ' with first factor: ', (first_factor[count] / len(self.testloaderdiseased)),
                  ' with second factor: ', (second_factor[count] / len(self.testloaderdiseased)), '\n')
            count += 1

        # print('For value of k: ', k, ' we have the maximum dice score of: ',max_k)

        # dice,first_factor,second_factor
        return max_k, max_first, max_second

    def check_anomaly_score_all_patients(self, model):

        self.model.eval()
        img_width = self.img_width
        img_height = self.img_height

        residual_volume = list()

        # mi calcolo tutti i residual e creo le due matrici di std e mean
        std_mask = np.zeros((img_height, img_width))

        for j, data in tqdm(enumerate(self.trainloader), total=int(len(self.trainset) / self.trainloader.batch_size)):
            img = data['image']
            img = img.cuda()
            reconstruction, loss, perplexity = model(img)

            for x in range(len(img)):
                rec = reconstruction[x, 0, :, :].cpu().detach().numpy()
                # display(Image.fromarray(np.uint8(np.squeeze(rec))))
                rec = rec.clip(min=0)
                original = img[x, 0, :, :].cpu()
                _, pa_bi = cv2.threshold(np.array(original), 0, 255, cv2.THRESH_BINARY)
                std_mask += pa_bi / 255
                diff = np.subtract(np.array(original), np.array(rec), dtype=np.float64)
                # diff = diff.clip(min=0)
                residual_volume.append(diff)

        std_mask[std_mask > 0] = 1
        pixels = list()
        std = np.zeros((img_height, img_width))
        mean = np.zeros((img_height, img_width))

        for i in tqdm(range(img_height)):
            for j in range(img_width):
                for index in range(len(residual_volume)):
                    pixels.append(residual_volume[index][i][j])
                    if len(pixels) == len(
                            residual_volume):  # se entro vuol dire ho fatto una lista dello stesso pixel di tutte le imm
                        numpy_pixels = np.array(pixels, dtype=np.float64)
                        std_0 = np.std(numpy_pixels, dtype=np.float64)
                        mean_0 = np.mean(numpy_pixels, dtype=np.float64)
                        std[i][j] = std_0
                        mean[i][j] = mean_0
                        pixels.clear()

        # used for lower and upper bound of statistic
        a = 0

        value = np.array([3])
        k_collection = np.zeros(np.shape(value)[0])
        first_factor = np.zeros(np.shape(value)[0])
        second_factor = np.zeros(np.shape(value)[0])

        f_factor = []

        # Liste per salvarmi tutti i fattori dei parametri

        # Create List of empty list
        for i in range(len(value)):
            f_factor.append([])

        s_factor = []

        # Create List of empty list
        for i in range(len(value)):
            s_factor.append([])
        images = list()

        total_dice_score = 0
        # print(len(self.testloadermask))
        # devi aggiungere anche testloader diseased!

        for i in tqdm(range(len(seg_images))):
            real_mask = seg_images[i][0, 0, :, :].cpu().detach().numpy()
            reconstruction, loss, perplexity = model(test_images[i].cuda())

            rec = np.array(reconstruction[0, 0, :, :].cpu().detach().numpy()) * 255
            rec = rec.clip(min=0)
            test_image = np.array(test_images[i][0, 0, :, :].cpu().detach().numpy()) * 255
            # print('Image number: ', i)

            diff = np.subtract(test_image, rec, dtype=np.float64)
            # diff = diff.clip(min=0)

            full_mask, resized_mask = create_images_masks(test_images[i][0, 0, :, :])
            count = 0

            for x in value:
                maps = find_anomalies(diff / 255, mean, std, x, img_width, img_height)
                # np.shape(maps) = 256,256 min max 0,255

                mu = (resized_mask / 255) * (maps)
                # np.shape(mu) = 256,256 min max 0,255

                intersection, blue_pixels = check_std(std=std_mask, anomaly_map=mu / 255)

                mu = intersection * 255

                dice_loss, a, b = new_dice(real_mask, mu / 255)
                k_collection[count] += dice_loss
                first_factor[count] += a
                second_factor[count] += b
                f_factor[count].append(a)
                s_factor[count].append(b)
                count += 1
                '''images.append(Image.fromarray(np.uint8(test_image)))
                images.append(Image.fromarray(np.uint8(rec)))
                images.append(Image.fromarray(np.uint8(diff)))
                images.append(Image.fromarray(np.uint8(real_mask*255)))
                images.append(Image.fromarray(np.uint8(Image.fromarray(np.uint8(maps)))))'''

                # print('Evaluated image with k = ',x)
                # plot_images(images)
                # showImagesHorizontally(images)
                # images.clear()

        count = 0
        count_max = 0
        max_k = 0
        k = 0
        max_first = 0
        max_second = 0

        for x in value:

            if (k_collection[count] / len(seg_images)) > max_k:
                max_k = (k_collection[count] / len(seg_images))
                max_first = (first_factor[count] / len(seg_images))
                max_second = (second_factor[count] / len(seg_images))
                count_max = count
                k = x

            print('For value of k: ', x,
                  ' the dice loss is: ', (k_collection[count] / len(seg_images)),
                  ' with first factor: ', (first_factor[count] / len(seg_images)),
                  ' with second factor: ', (second_factor[count] / len(seg_images)), '\n')
            count += 1

        # print('For value of k: ', k, ' we have the maximum dice score of: ',max_k)

        # dice,first_factor,second_factor
        return max_k, max_first, max_second

    # alleno autoencoder controllando il dice score ogni tot epoche per evitare la funzione identità
    def start1(self):
        # summary(self.model, self.input_shape)
        import copy

        patients_image_folders_path = get_patients_images_path(r'/kaggle/input/11-paz-norm-centred-crop')

        epoch_patients_before_ce = 0
        last_loss = 100000
        patience = self.early_stopping
        patience_dice = self.patience_dice
        trigger_times = 0
        trigger_times_dice = 0
        dice = -1
        last_dice = -1
        dice_val = 0

        best_weight_par = self.model.state_dict()
        model_clone = copy.deepcopy(self.model)
        model_clone.load_state_dict(best_weight_par)
        model1 = model_clone

        training_info = TrainingInfo()
        best_epoch = 0
        count_shuffle = 0
        last_dice = 0
        loss_vq = []
        loss_ce = []
        loss_tot = []

        best_single_mean_distribution = []
        best_single_std_distribution = []
        single_mean_distribution = []
        single_std_distribution = []

        for epoch in range(self.epochs):

            count_shuffle = 0
            print(f"Epoch {epoch + 1} of {self.epochs}")

            '''if 1:
                train_epoch_loss = self.train_step_norm(epoch)
                loss_tot.append(train_epoch_loss)
                loss_vq.append(train_epoch_loss)
                loss_ce.append(1)
            else:'''

            train_epoch_loss, vq = self.train_step(epoch)
            loss_tot.append(train_epoch_loss)

            self.scheduler.step(train_epoch_loss)

            print(self.optimizer.state_dict()['param_groups'][0]['lr'])

            print(f"Train Loss: {train_epoch_loss * 1000:.4f}")

            current_loss = train_epoch_loss

            '''#early stopping
            if current_loss > last_loss:
                trigger_times += 1
                print('Trigger Times Validation:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping! TRAINING COMPLETE')
                    print('Loading model at epoch ',best_epoch, ' with a dice of: ',last_dice, ' and a validation loss of: ',last_loss*1000)

                    return model1,loss_tot,loss_vq,loss_ce,best_single_mean_distribution,best_single_std_distribution

            else:
                print('trigger times: 0')
                trigger_times = 0
                last_loss = current_loss
                best_weight_par = self.model.state_dict()
                model_clone = copy.deepcopy(self.model)
                model_clone.load_state_dict(best_weight_par)
                best_epoch = epoch
                model1 = model_clone'''

            ### da qui i poi crea funzione che parte ogni 5 epoche

            value = np.array([2])

            # Liste per salvarmi tutti i fattori dei parametri
            f_factor = []
            # Create List of empty list
            for i in range(len(value)):
                f_factor.append([])

            s_factor = []

            # Create List of empty list
            for i in range(len(value)):
                s_factor.append([])

            images = list()

            total_dice_score = 0

            transform = transforms.Compose([
                transforms.Resize((self.img_width, self.img_width)),
                transforms.ToTensor()])

            k_collection = np.zeros(np.shape(value)[0])
            first_factor = np.zeros(np.shape(value)[0])
            second_factor = np.zeros(np.shape(value)[0])

            f_factor = []

            # Liste per salvarmi tutti i fattori dei parametri
            for i in range(len(value)):
                f_factor.append([])

            s_factor = []

            # Create List of empty list
            for i in range(len(value)):
                s_factor.append([])

            dice_equal_to_zero = []

            # Create List of empty list
            for i in range(len(value)):
                dice_equal_to_zero.append(0)

            print('trigger times Dice:', trigger_times_dice)

            if epoch % 5 == 0 and epoch > 25:
                tot_dice = 0
                tot_first = 0
                tot_second = 0
                single_mean_distribution.clear()
                single_std_distribution.clear()
                count = 0

                for h in tqdm(range(len(patients_image_folders_path))):
                    count_dice_uguale_a_zero = 0
                    health = patients_image_folders_path[h] + '/brainmetshare/metshare/train/healthy/id/2'
                    disease = patients_image_folders_path[h] + '/brainmetshare/metshare/test/disease/id/2'
                    seg = patients_image_folders_path[h] + '/brainmetshare/metshare/test/disease/id/seg'
                    # healthy
                    h = sorted(glob.glob(health + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))
                    # disease
                    d = sorted(glob.glob(disease + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))
                    # seg
                    s = sorted(glob.glob(seg + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))
                    std_mask = np.zeros((self.img_width, self.img_width))
                    count_batch = 0
                    for img in h:
                        im = Image.open(img)
                        im = transform(im)
                        std_mask += np.array(np.squeeze(im))
                        im = torch.unsqueeze(im, 0)
                        im = im.to(self.device)
                        reconstruction, r_loss, perplexity = self.model(im)
                        reconstruction = np.squeeze(reconstruction)
                        reconstruction[reconstruction < 4 / 255] = 0

                        original = np.squeeze(im)
                        diff = np.subtract(original.cpu(), np.array(reconstruction.cpu().detach().numpy()))
                        diff = np.array(diff).flatten()
                        diff = diff[diff != 0]
                        if count_batch == 0:
                            flat_residual_distribution = np.copy(diff)
                        else:
                            flat_residual_distribution = np.concatenate((flat_residual_distribution, diff), axis=0)
                        count_batch += 1
                        # print('Len flat_residual_distribution', len(flat_residual_distribution))

                    for img in d:
                        im = Image.open(img)
                        im = transform(im)
                        im = torch.unsqueeze(im, 0)
                        std_mask += np.array(np.squeeze(im))
                        im = im.to(self.device)
                        reconstruction, r_loss, perplexity = self.model(im)
                        reconstruction[reconstruction < 4 / 255] = 0
                        reconstruction = np.squeeze(reconstruction)
                        original = np.squeeze(im)
                        diff = np.subtract(original.cpu(), reconstruction.cpu().detach().numpy())
                        diff = np.array(diff).flatten()
                        diff = diff[diff != 0]
                        flat_residual_distribution = np.concatenate((flat_residual_distribution, diff), axis=0)
                        # print('Len flat_residual_distribution', len(flat_residual_distribution))

                    single_mean = np.mean(flat_residual_distribution)
                    single_std = np.std(flat_residual_distribution)
                    single_mean_distribution.append(single_mean)
                    single_std_distribution.append(single_std)
                    std_mask[std_mask > 0] = 1
                    # display(Image.fromarray(np.uint8(std_mask*255)))

                    # calcolo dice di imm malate

                    for y in range(len(d)):

                        im = Image.open(d[y])
                        im = transform(im)
                        im = torch.unsqueeze(im, 0)
                        # print('Shape im',np.shape(im))
                        # print('Shape im[0, 0, :, :] ',np.shape(im[0, 0, :, :]))
                        im = im.to(self.device)

                        se = Image.open(s[y])
                        se = transform(se)
                        se = np.squeeze(se)

                        reconstruction, r_loss, perplexity = self.model(im)

                        reconstruction = np.squeeze(reconstruction.cpu().detach().numpy())
                        diff = np.subtract(im[0, 0, :, :].cpu() * 255, np.array(reconstruction) * 255)

                        full_mask, resized_mask = create_images_masks(im[0, 0, :, :].cpu())
                        # print('resize mask')

                        # display(Image.fromarray(np.uint8(resized_mask)))
                        count_coll = 0

                        # for x in value:
                        for x in range(len(value)):
                            # print(x)
                            maps = find_anomalies_single_value(diff / 255, single_mean, single_std, value[x],
                                                               self.img_width)
                            # print('map')
                            # display(Image.fromarray(np.uint8(maps)))
                            mu = (resized_mask / 255) * (maps)
                            # print('Mu')
                            # display(Image.fromarray(np.uint8(np.squeeze(mu))))

                            mu = std_mask * mu
                            # print('Mu')
                            # display(Image.fromarray(np.uint8(np.squeeze(mu))))

                            dice_loss, a, b = new_dice(se, mu / 255)
                            if dice_loss == 0: dice_equal_to_zero[x] += 1
                            k_collection[count_coll] += dice_loss
                            first_factor[count_coll] += a
                            second_factor[count_coll] += b
                            f_factor[count_coll].append(a)
                            s_factor[count_coll].append(b)
                            count_coll += 1
                            # print('dice',dice_loss,' first factor',a,' second factor', b)

                            '''tot_dice += dice_loss
                            tot_first += a
                            tot_second += b'''

                        count += 1
                        count_coll = 0

                max_k = 0
                k = 0
                count_coll
                for x in value:
                    if (k_collection[count_coll] / count) > max_k:
                        max_k = (k_collection[count_coll] / count)
                        k = x

                    print('For value of k: ', x,
                          ' the dice loss is: ', (k_collection[count_coll] / count),
                          ' with first factor: ', (first_factor[count_coll] / count),
                          ' with second factor: ', (second_factor[count_coll] / count),
                          'missed anomalies', dice_equal_to_zero[count_coll], 'n')
                    count_coll += 1

                '''print('Number of images of test set: ', count)
                print('For value of k: 3',
                  ' the dice loss is: ', (tot_dice/count),
                  ' with first factor: ', (tot_first/count),
                  ' with second factor: ', (tot_second/count),'\n')'''
                count_coll -= 1
                dice = k_collection[count_coll] / count

                # self.check_anomaly_score_all_patients(self.model)
                # dice,first_factor,second_factor = self.check_anomaly_score(self.model)
                # dice,first_factor,second_factor = self.check_anomaly_score_all_patients(self.model)

                # print(f'Dice score is: {dice:.7f}')

                if dice > last_dice:
                    last_dice = dice
                    trigger_times_dice = 0
                    # salvo copia modello migliore
                    best_weight_par = self.model.state_dict()
                    model_clone = copy.deepcopy(self.model)
                    model_clone.load_state_dict(best_weight_par)
                    model1 = model_clone
                    best_epoch = epoch
                    dice_val = last_loss
                    best_single_mean_distribution = single_mean_distribution.copy()
                    best_single_std_distribution = single_std_distribution.copy()

                else:
                    trigger_times_dice += 1
                    if trigger_times_dice > patience_dice:
                        print('Early stopping due to Dice. TRAINING COMPLETE')
                        print('Loading model at epoch ', best_epoch, ' with a dice of: ', last_dice,
                              ' and a validation loss of: ', last_loss * 1000)
                        return model1, loss_tot, loss_vq, loss_ce, best_single_mean_distribution, best_single_std_distribution

            print(f'Dice score is: {dice:.4f}')
            print(f'Best Dice score is: {last_dice:.4f}')
            print(f"Last Loss: {last_loss * 1000:.4f}")
            print(f'shuffle_count: {count_shuffle}')

        print('TRAINING COMPLETE')
        print('Loading model at epoch ', best_epoch, ' with a dice of: ', last_dice, ' and a validation loss of: ',
              last_loss * 1000)

        return model1, loss_tot, loss_vq, loss_ce, best_single_mean_distribution, best_single_std_distribution

    # alleno autoencoder senza controllare dice
    def start(self):
        # summary(self.model, self.input_shape)

        # early stopping
        last_loss = 100000
        patience = self.early_stopping
        triggertimes = 0

        training_info = TrainingInfo()

        for epoch in range(self.epochs):

            print(f"Epoch {epoch + 1} of {self.epochs}")
            train_epoch_loss = self.train_step()
            valid_epoch_loss, recon_images, original_images = self.validate_step()
            self.scheduler.step(valid_epoch_loss)
            print(self.optimizer.state_dict()['param_groups'][0]['lr'])

            '''tensor = original_images.squeeze(0)
            toImageFromTensorTrasform = transforms.ToPILImage()
            img = toImageFromTensorTrasform(tensor)
            display(img)
            %pylab inline
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            imgplot = plt.imshow(img)
            plt.show()'''

            epoch_info = EpochInfo(model=self.model,
                                   epoch=epoch,
                                   recon_images=recon_images,
                                   original_image=original_images,
                                   train_epoch_loss=train_epoch_loss,
                                   valid_epoch_loss=valid_epoch_loss)

            epoch_info.save()
            training_info.add_grid_image(recon_images=recon_images)
            training_info.add_epoch_info(epoch_info=epoch_info)

            print(f"Train Loss: {train_epoch_loss * 1000:.4f}")
            print(f"Val Loss: {valid_epoch_loss * 1000:.4f}")

            current_loss = valid_epoch_loss

            # early stopping
            if current_loss > last_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    training_info.save_validation_gif_results()
                    training_info.save_loss_plot()
                    print('Early stopping! TRAINING COMPLETE')
                    return self.model, training_info.best_epoch, training_info.best_val_loss


            else:
                print('trigger times: 0')
                trigger_times = 0

            if current_loss < last_loss:
                last_loss = current_loss

            print(f"Last Loss: {last_loss * 1000:.4f}")

        training_info.save_validation_gif_results()
        training_info.save_loss_plot()
        print('TRAINING COMPLETE')

        return self.model, training_info.best_epoch, training_info.best_val_loss

    def train_step(self, epoch):
        self.model.train()
        running_loss = 0.0
        counter = 0
        loss_vq = []

        for i in tqdm(range(len(self.training_set_per_shuffle))):
            counter += 1
            data = self.training_set_per_shuffle[i]
            data = data.to(self.device)

            self.optimizer.zero_grad()
            reconstruction, r_loss, perplexity = self.model(data)

            recon_error = F.mse_loss(reconstruction, data)
            rec_loss_vqvae = recon_error + r_loss

            loss = rec_loss_vqvae

            loss_vq.append(rec_loss_vqvae.cpu().detach().numpy())
            loss.backward()

            running_loss += loss.item()
            self.optimizer.step()

        train_loss = running_loss / (counter * self.trainloader.batch_size)

        return train_loss, np.mean(np.sum(loss_vq))

    # lo uso per CE-VQVAE
    def train_step_patches(self, epoch):
        self.model.train()
        running_loss = 0.0
        counter = 0
        lamda = 0.5
        loss_vq = []
        loss_ce = []

        for i in tqdm(range(len(self.training_set_per_shuffle))):
            # for i, data in tqdm(enumerate(self.trainloader), total=int(len(self.trainset) / self.trainloader.batch_size)):
            counter += 1
            data = self.training_set_per_shuffle[i]

            # cp_data = np.array(np.copy(np.squeeze(data)))

            # cp_data[cp_data>0] = 1

            # se non ho patches nere ma griglia
            # masked_data = self.masked_data

            # se ho patches nere
            masked_data = self.masked_data[i]

            data = data.to(self.device)

            self.optimizer.zero_grad()
            reconstruction, r_loss, perplexity = self.model(data)
            # reconstruction[0,0,:,:] = cp_data * reconstruction[0,0,:,:]

            recon_error = F.mse_loss(reconstruction, data)
            rec_loss_vqvae = recon_error + r_loss

            rec_loss_ce = 0

            #################
            masked_data = []
            original_image = np.array(np.squeeze(data.cpu()))
            maschera = np.zeros((self.img_width, self.img_width))
            # show_grey(original_image)
            count = np.nonzero(original_image)
            min_x = np.min(count[0])
            min_y = np.min(count[1])
            max_x = np.max(count[0])
            max_y = np.max(count[1])

            thirty_per_cent_x = int(((max_x - min_x) / 100) * 80)
            thirty_per_cent_y = int(((max_y - min_y) / 100) * 80)

            maschera[min_x:max_x, min_y:max_y] = 255

            rec_loss_ce = 0
            num_of_patches = 0
            for i in range(int((max_x - min_x) / thirty_per_cent_x)):
                start_x = min_x + i * thirty_per_cent_x
                for j in range(int((max_y - min_y) / thirty_per_cent_y)):
                    or_copy = np.copy(original_image)
                    test_patch = np.copy(original_image)

                    start_y = min_y + j * thirty_per_cent_y
                    or_copy[start_x: start_x + thirty_per_cent_x, start_y: start_y + thirty_per_cent_y] = 0

                    threshold_patch = np.copy(original_image)
                    threshold_patch[start_x: start_x + thirty_per_cent_x, start_y: start_y + thirty_per_cent_y] = -1
                    threshold_patch[threshold_patch > -1] = 0
                    threshold_patch[threshold_patch == -1] = 1
                    number_of_white_pix_patch = np.sum(threshold_patch == 1)
                    test_patch = test_patch * threshold_patch
                    number_of_white_pix_test_patch = np.sum(test_patch > 0)

                    if number_of_white_pix_test_patch > (number_of_white_pix_patch / 100) * 30:
                        # display(Image.fromarray(np.uint8(or_copy*255)))
                        or_copy = torch.from_numpy(or_copy)
                        or_copy = torch.unsqueeze(or_copy, dim=0)
                        or_copy = torch.unsqueeze(or_copy, dim=0)
                        masked_data_0 = or_copy.to(self.device, dtype=torch.float)
                        rec_ce, r_loss, _ = self.model(masked_data_0)
                        # rec_ce[0,0,:,:] = cp_data * rec_ce[0,0,:,:]
                        recon_error = F.mse_loss(rec_ce, data)
                        rec_loss_ce += recon_error + r_loss
                        num_of_patches += 1

            rec_loss_ce = rec_loss_ce / num_of_patches

            # for x in range(len(data)):
            # if epoch % 50 == 0:

            loss = (1 - lamda) * rec_loss_vqvae + lamda * rec_loss_ce
            loss_ce.append(rec_loss_ce.cpu().detach().numpy())
            loss_vq.append(rec_loss_vqvae.cpu().detach().numpy())
            loss.backward()
            # loss.mean().backward()
            running_loss += loss.item()
            self.optimizer.step()
        train_loss = running_loss / (counter * self.trainloader.batch_size)
        # display(Image.fromarray(np.uint8(np.squeeze(reconstruction[0].cpu().detach().numpy())*255)))

        # display(Image.fromarray(np.uint8(np.squeeze(rec_ce[0].cpu().detach().numpy())*255)))
        return train_loss, np.mean(np.sum(loss_vq)), np.mean(np.sum(loss_ce))

    def train_step_norm(self, epoch):
        self.model.train()
        running_loss = 0.0
        counter = 0

        for i in tqdm(range(len(self.training_set_per_shuffle))):
            # for i, data in tqdm(enumerate(self.trainloader), total=int(len(self.trainset) / self.trainloader.batch_size)):
            counter += 1
            data = self.training_set_per_shuffle[i]
            # print(np.shape(masked_data))
            # display(Image.fromarray(np.uint8(np.squeeze(data)*255)))

            # display(Image.fromarray(np.uint8(np.squeeze(masked_data)*255)))
            '''for x in range(len(data)):
            display(Image.fromarray(np.uint8(np.squeeze(data[x])*255)))

            display(Image.fromarray(np.uint8(np.squeeze(masked_data[x])*255)))'''

            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, r_loss, perplexity = self.model(data)
            recon_error = F.mse_loss(reconstruction, data)
            loss = recon_error + r_loss
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
        train_loss = running_loss / (counter * self.trainloader.batch_size)
        return train_loss

    def validate_step(self):
        self.model.eval()
        running_loss = 0.0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.testloader), total=int(len(self.testset) / self.testloader.batch_size)):
                counter += 1
                data = data['image']
                '''tensor = data.squeeze(0)
                toImageFromTensorTrasform = transforms.ToPILImage()
                img = toImageFromTensorTrasform(tensor)
                display(img)'''
                data = data.to(self.device)
                reconstruction, loss, perplexity = self.model(data)
                # bce_loss = self.criterion(reconstruction, data)
                # loss = self.compute_loss(bce_loss, mu, logvar)
                recon_error = F.mse_loss(reconstruction, data)
                loss = recon_error + loss
                running_loss += loss.item()

                # save the last batch input and output of every epoch
                if i == int(len(self.testloader) / self.testloader.batch_size) - 1:
                    recon_images = reconstruction
                    original_images = data
        val_loss = running_loss / counter
        return val_loss, recon_images, original_images

    @staticmethod
    def final_loss(bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: reconstruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        bce = bce_loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld


# funzione per allenare un autoencodersu una singola immagine per verificare funzione identità
def train_single_image(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
                       commitment_cost, decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((196, 196)),
        transforms.ToTensor()])
    # train imm sana
    img = PIL.Image.open(
        "../input/d/giuseppepaolini/11-paz-norm-centred-crop/Mets_089/brainmetshare/metshare/train/healthy/id/2/106.png").convert(
        'L')
    img = transform(img)
    img = img[None, :]
    # train imm malata
    img1 = PIL.Image.open(
        "../input/d/giuseppepaolini/11-paz-norm-centred-crop/Mets_148/brainmetshare/metshare/test/disease/id/2/094.png").convert(
        'L')
    img2 = img1.copy()

    img1 = transform(img1)
    img1 = torch.cat([img1, img1, img1, img1, img1, img1, img1, img1], dim=0)
    # img1 = np.array([img1,img1,img1,img1,img1,img1,img1,img1])

    img1 = img1[None, :]

    im = list()
    im.append(img)
    img2 = PIL.Image.open(
        "../input/d/giuseppepaolini/11-paz-norm-centred-crop/Mets_148/brainmetshare/metshare/test/disease/id/2/094.png").convert(
        'L')
    img2 = transform(img2)
    img2 = img2[None, :]

    im.append(img2)

    '''original = Image.fromarray(np.int8(np.squeeze(img1) * 255))

    original.show()'''

    # img1 = img[None, :]

    decay = 0
    # initialize the model
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
                       commitment_cost, decay).to(device)

    # Definition of the stochastic optimizer used to train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    epoche = 10000

    best_loss = 99999999999999999999
    early_stopping = 0
    patient = 20000
    for epoch in tqdm(range(epoche)):

        # print(f"Epoch {epoch + 1} of {epoche}")
        running_loss = 0.0

        img = im[0].to(device)
        # Zero the gradient
        optimizer.zero_grad()

        reconstruction, loss, perplexity = model(img)
        recon_error = F.mse_loss(reconstruction, img)
        loss = recon_error + loss

        # print(loss)
        loss.backward()

        # aggiorno pesi
        optimizer.step()

        running_loss += loss.item()  # gradiente

        train_loss = running_loss / (8)
        train_epoch_loss = train_loss
        if train_loss > best_loss:
            early_stopping += 1
            if early_stopping > patient:
                break
        else:
            best_loss = train_loss
            early_stopping = 0
            model1 = model
            # torch.save(model.state_dict(), r'C:\Users\user\Desktop\model-13')

        # show_image_tensor(original_images)
        # print(f"Train Loss: {train_epoch_loss * 1000:.4f}")
        # print(f"Best loss: {best_loss * 1000:.4f}")
        # print(f"Trigger: {early_stopping}")

    print('TRAINING COMPLETE')
    print(f"Train Loss: {train_epoch_loss * 1000:.4f}")
    print('Loading best model')

    model.eval()

    reconstruction, loss, perplexity = model(im[0].to(device))

    print('Reconstruction')
    img1_arr = reconstruction[0, 0, :, :].to(device).cpu().detach().numpy() * 255
    img1_arr[img1_arr < 0] = 0

    ima = Image.fromarray(np.int8(img1_arr))
    plt.imshow(ima)
    # plt.savefig(f'/kaggle/working/outputs/R.jpg')

    plt.figure()
    reconstruction, loss, perplexity = model(im[1].to(device))
    # ima.save("R")

    print('Test Image')
    # img1 = Image.fromarray(np.int8(np.squeeze(img1)*255))
    # print(np.shape(np.squeeze(im[1])))
    ima = Image.fromarray(np.int8(np.squeeze(im[1]) * 255))
    # ima[ima<0] = 0
    plt.imshow(ima)
    plt.figure()

    print('Test Image reconstructed')

    img1_arr = reconstruction[0, 0, :, :].to(device).cpu().detach().numpy() * 255
    img1_arr[img1_arr < 0] = 0
    ima = Image.fromarray(np.int8(img1_arr))
    plt.imshow(ima)
    plt.figure()

    # plt.savefig(f'/kaggle/working/outputs/T.jpg')

    # ima.save("T")


torch.manual_seed(0)


# alleno più autoencoders seguendo il piano sperimentale di placket burman per fine tuning dei parametri
def train_mode_burman(plane, burman, different_attempt):
    # epochs     lr    batch  weight  com_cost  in_size  lr_pat   lr_fact     em_dim  num_em    early_stopping
    sperimental_plane = [[1000, 1e-3, 8, 0, 0.25, 128, 75, 0.01, 128, 256, 100],
                         [1500, 1e-2, 24, 1e-5, 0.5, 256, 99, 0.1, 256, 512, 150]]

    '''[1,	      0,	1,   	0,	       1,	     1,	      1,	  0,	       0,	   0,	   0],        
                        [0,        0,   1 ,  	1,	       1,	     0,	      0,	   0,	      1,	   0,	   1],
                        [0,        0,    1,   	0,	       0,	     1,	      0,	   1,	      1,	   1,	    0]'''

    # epochs     lr    batch  weight  com_cost  in_size  lr_pat   lr_fact  em_dim  num_em  early_stopping
    # num_hiddens,num_residual,num residual layer
    different_attempt = [[700, 1e-3, 8, 0, 0.45, 256, 70, 0.1, 128, 512, 100, 256, 32, 2]]

    plackett_burman = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                       [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
                       [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                       [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
                       [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                       [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                       [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
                       [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]]

    '''for i in range(len(burman)):

        #os.chdir("/kaggle/input/168images")

        epochs = plane[1*burman[i][0]][0]
        learning_rate = plane[1*burman[i][1]][1]
        batch_size = plane[1*burman[i][2]][2]
        weight_decay = plane[1*burman[i][3]][3]
        commitment_cost = plane[1*burman[i][4]][4]
        img_width = plane[1*burman[i][5]][5]
        lr_patience = plane[1*burman[i][6]][6]
        lr_factor = plane[1*burman[i][7]][7]
        embedding_dim = plane[1*burman[i][8]][8]
        num_embeddings = plane[1*burman[i][9]][9]
        early_stopping = plane[1*burman[i][10]][10]
        img_height = img_width'''


for models_num in range(np.shape(different_attempt)[0]):

    # os.chdir("/kaggle/input/168images")

    epochs = different_attempt[models_num][0]
    learning_rate = different_attempt[models_num][1]
    batch_size = different_attempt[models_num][2]
    weight_decay = different_attempt[models_num][3]
    commitment_cost = different_attempt[models_num][4]
    img_width = different_attempt[models_num][5]
    lr_patience = different_attempt[models_num][6]
    lr_factor = different_attempt[models_num][7]
    embedding_dim = different_attempt[models_num][8]
    num_embeddings = different_attempt[models_num][9]
    early_stopping = different_attempt[models_num][10]
    img_height = img_width
    num_hiddens = different_attempt[models_num][11]  # larghezza latent space 512/1024
    num_residual_hiddens = different_attempt[models_num][12]
    num_residual_layers = different_attempt[models_num][13]

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # DATASET GENERATION

    brain = 'metshare'
    plantVillageTrain, plantVillageVal, plantVillageTestHealthy, plantVillageTestDiseased, plantVillageTestDiseasedMaskSeg = generateDataset(
        brain, transform)

    trainloader = get_training_dataloader(plantVillageTrain, batch_size)
    validationloader = get_validation_dataloader(plantVillageVal, batch_size=1)
    testloaderDiseased = get_test_dataloader(plantVillageTestDiseased, batch_size=1)
    testloaderHealthy = get_test_dataloader(plantVillageTestHealthy, batch_size=1)
    testloaderDiseasedMask = get_test_dataloader(plantVillageTestDiseasedMaskSeg, batch_size=1)

    input_channel = 1

    decay = 0
    # initialize the model
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay).to(device)

    criterion = nn.MSELoss(reduction='sum')

    optimizer = opt.Adam(model.parameters(), lr=learning_rate, amsgrad=False, weight_decay=weight_decay)
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    compute_loss = lambda a, b, c: a
    input_shape = (input_channel, img_width, img_height)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=lr_factor, patience=lr_patience, threshold=0.0001,
                                                           threshold_mode='abs', verbose=True)
    ddp_model = torch.nn.DataParallel(model)

    engine = Engine(model=model,
                    trainloader=trainloader,
                    trainset=plantVillageTrain,
                    testloader=validationloader,
                    testset=plantVillageVal,
                    epochs=epochs,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    img_width=img_width,
                    img_height=img_height,
                    compute_loss=compute_loss,
                    device=device,
                    early_stopping=early_stopping)

    print('Training model ' + str(models_num + 1))

    if TRAIN:
        model, best_epoch, best_loss = engine.start()

        if LOAD_BEST_MODEL:
            print('Loading epoch #{}'.format(best_epoch))
            load_model(model, best_epoch)
            torch.save(model.state_dict(),
                       f"/kaggle/working/outputs/best_param/best_model")  # SAVE BEST MODEL ON DIFFERENT DIRECTORY
        else:
            print('Using net as-is')
    else:
        load_model(model, PARAMS_TO_LOAD)

    '''engine.segmentation_performance_computation_disease(model, testloaderDiseased, plantVillageTestDiseased,
                                                        testloaderDiseasedMask, plantVillageTestDiseasedMaskSeg)'''

    best_loss = best_loss * 1000
    best_loss = f'{best_loss:.5f}'[:-1]

    # calculate mean and std of residuals

    residual_volume = list()

    # mi calcolo tutti i residual e creo le due matrici di std e mean

    for j, data in tqdm(enumerate(trainloader), total=int(len(plantVillageTrain) / trainloader.batch_size)):
        img = data['image']
        img = img.cuda()
        reconstruction, loss, perplexity = model(img)

        for x in range(trainloader.batch_size):
            rec = reconstruction[x, 0, :, :].cpu().detach().numpy()
            rec = rec.clip(min=0)
            original = img[x, 0, :, :].cpu()
            diff = np.subtract(np.array(original), np.array(rec), dtype=np.float64)
            # diff = diff.clip(min=0)
            residual_volume.append(diff)

    pixels = list()
    std = np.zeros((img_height, img_width))
    mean = np.zeros((img_height, img_width))

    for i in tqdm(range(img_height)):
        for j in range(img_width):
            for index in range(len(residual_volume)):
                pixels.append(residual_volume[index][i][j])
                if len(pixels) == len(
                        residual_volume):  # se entro vuol dire ho fatto una lista dello stesso pixel di tutte le imm
                    numpy_pixels = np.array(pixels, dtype=np.float64)
                    std_0 = np.std(numpy_pixels, dtype=np.float64)
                    mean_0 = np.mean(numpy_pixels, dtype=np.float64)
                    std[i][j] = std_0
                    mean[i][j] = mean_0
                    pixels.clear()

    from matplotlib.pyplot import figure, imshow, axis
    from matplotlib.image import imread

    # used for lower and upper bound of statistic

    a = 0

    value = np.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5])
    k_collection = np.zeros(np.shape(value)[0])
    first_factor = np.zeros(np.shape(value)[0])
    second_factor = np.zeros(np.shape(value)[0])

    f_factor = []

    # Liste per salvarmi tutti i fattori dei parametri

    # Create List of empty list
    for i in range(len(value)):
        f_factor.append([])

    s_factor = []

    # Create List of empty list
    for i in range(len(value)):
        s_factor.append([])
    images = list()

    total_dice_score = 0
    print(len(testloaderDiseasedMask))
    for item1, item2 in zip(enumerate(testloaderDiseased), enumerate(testloaderDiseasedMask)):

        i, data = item1
        u, data_seg = item2

        img = data['image'].cuda()
        seg = data_seg['image'].cuda()

        real_mask = seg[0, 0, :, :].cpu().detach().numpy()

        reconstruction, loss, perplexity = model(img)

        rec = np.array(reconstruction[0, 0, :, :].cpu().detach().numpy()) * 255
        rec = rec.clip(min=0)

        test_image = np.array(img[0, 0, :, :].cpu().detach().numpy()) * 255

        print('Image number: ', i)

        '''print('Original Image')
        display(Image.fromarray(np.uint8(test_image)))'''

        '''print('Reconstructed Image')
        display(Image.fromarray(np.uint8(rec)))'''

        diff = np.subtract(test_image, rec, dtype=np.float64)
        # diff = diff.clip(min=0)

        '''print('Residual Image')
        display(Image.fromarray(np.uint8(diff)))'''

        full_mask, resized_mask = create_images_masks(img[0, 0, :, :].cpu())

        '''print('Evaluated image mask')
        display(Image.fromarray(np.uint8(resized_mask)))'''

        count = 0
        for x in value:
            maps = find_anomalies(diff / 255, mean, std, x, img_width, img_height)
            mu = ((np.bitwise_and((resized_mask / 255).astype(int), (maps / 255).astype(int))) * 255).astype(int)
            dice_loss, a, b = new_dice(real_mask, mu / 255)
            k_collection[count] += dice_loss
            first_factor[count] += a
            second_factor[count] += b
            f_factor[count].append(a)
            s_factor[count].append(b)
            count += 1
            '''images.append(Image.fromarray(np.uint8(test_image)))
            images.append(Image.fromarray(np.uint8(rec)))
            images.append(Image.fromarray(np.uint8(diff)))
            images.append(Image.fromarray(np.uint8(real_mask*255)))
            images.append(Image.fromarray(np.uint8(Image.fromarray(np.uint8(maps)))))'''

            # print('Evaluated image with k = ',x)
            # plot_images(images)
            # showImagesHorizontally(images)
            # images.clear()

    count = 0
    count_max = 0
    max_k = 0
    k = 0
    for x in value:

        if (k_collection[count] / len(testloaderDiseased)) > max_k:
            max_k = (k_collection[count] / len(testloaderDiseased))
            count_max = count
            k = x

        print('For value of k: ', x,
              ' the dice loss is: ', (k_collection[count] / len(testloaderDiseased)),
              ' with first factor: ', (first_factor[count] / len(testloaderDiseased)),
              ' with second factor: ', (second_factor[count] / len(testloaderDiseased)), '\n')

        count += 1

    print('For value of k: ', k, ' we have the maximum dice score of: ', max_k)

    anomaly_mask_images = list()
    anomaly_mask_images.clear()

    # per mostrare le immagini
    for item1, item2 in zip(enumerate(testloaderDiseased), enumerate(testloaderDiseasedMask)):
        i, data = item1
        u, data_seg = item2

        img = data['image'].cuda()
        seg = data_seg['image'].cuda()

        real_mask = seg[0, 0, :, :].cpu().detach().numpy()

        reconstruction, loss, perplexity = model(img)

        rec = np.array(reconstruction[0, 0, :, :].cpu().detach().numpy()) * 255
        rec = rec.clip(min=0)

        test_image = np.array(img[0, 0, :, :].cpu().detach().numpy()) * 255

        print('Image number: ', i)

        diff = np.subtract(test_image, rec, dtype=np.float64)
        # diff = diff.clip(min=0)

        full_mask, resized_mask = create_images_masks(img[0, 0, :, :].cpu())

        images.clear()
        maps = find_anomalies(diff / 255, mean, std, k, img_width, img_height)
        mu = ((np.bitwise_and((resized_mask / 255).astype(int), (maps / 255).astype(int))) * 255).astype(int)
        dice_loss, a, b = new_dice(real_mask, mu / 255)

        images.append(Image.fromarray(np.uint8(test_image)))
        images.append(Image.fromarray(np.uint8(rec)))
        images.append(Image.fromarray(np.uint8(diff)))
        images.append(Image.fromarray(np.uint8(real_mask * 255)))
        images.append(Image.fromarray(np.uint8(Image.fromarray(np.uint8(maps)))))
        anomaly_mask_images.append((Image.fromarray(np.uint8(Image.fromarray(np.uint8(maps))))))

        print('Evaluated image with k = ', k)
        showImagesHorizontally(images)

    model_path = '/kaggle/working/all/Model-' + str(models_num) + '_with_Val_loss_' + best_loss + '_dice_' + str(
        max_k)

    # creo cartella modello
    os.mkdir(model_path)

    # creo cartella per anomaly mask images
    os.mkdir(model_path + '/anomaly_mask')

    # la imposto come current
    os.chdir(model_path + '/anomaly_mask')

    for n in range(len(anomaly_mask_images)):
        anomaly_mask_images[n].save("Anomaly_mask_" + str(n) + ".png")

    # imposto di nuovo quella di default
    os.chdir("/kaggle/input/168images")

    dir_path = f"/kaggle/working/outputs/params"
    shutil.rmtree(dir_path, ignore_errors=True)

    dir_path = f"/kaggle/working/outputs/images"
    shutil.rmtree(dir_path, ignore_errors=True)

    shutil.move(f"/kaggle/working/outputs",
                '/kaggle/working/all/Model-' + str(models_num) + '_with_Val_loss_' + best_loss + '_dice_' + str(
                    max_k))

    dir_path = f"/kaggle/working/outputs/images"
    shutil.rmtree(dir_path, ignore_errors=True)

    dir_path = f"/kaggle/working/outputs/test"
    shutil.rmtree(dir_path, ignore_errors=True)

    dir_path = f"/kaggle/working/outputs/best_param"
    shutil.rmtree(dir_path, ignore_errors=True)

    dir_path = f"/kaggle/working/outputs"
    shutil.rmtree(dir_path, ignore_errors=True)

    # os.mkdir('/kaggle/working/outputs/all')
    os.mkdir('/kaggle/working/outputs')
    os.mkdir('/kaggle/working/outputs/images')
    os.mkdir('/kaggle/working/outputs/params')
    os.mkdir('/kaggle/working/outputs/test')
    os.mkdir('/kaggle/working/outputs/best_param')

    # move the test images from the images directory to the test directory


##################################
# questa parte di codice serve per allenare singoli autoencoders su singoli pazienti

def get_real_anomaly_seg(seg_images, test_images, path_dataset, img_width, img_height):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()])

    brain = 'metshare'
    plantVillageTrain, plantVillageTestHealthy, plantVillageTestDiseased, \
    plantVillageTestDiseasedMaskSeg = generateDataset(
        brain, transform)

    testloaderDiseasedMask = get_test_dataloader(plantVillageTestDiseasedMaskSeg, batch_size=1)
    testloaderDiseased = get_test_dataloader(plantVillageTestDiseased, batch_size=1)

    for i, data in tqdm(enumerate(testloaderDiseasedMask),
                        total=int(len(plantVillageTestDiseasedMaskSeg) / testloaderDiseasedMask.batch_size)):
        img = data['image']
        seg_images.append(img)

    for i, data in tqdm(enumerate(testloaderDiseased),
                        total=int(len(plantVillageTestDiseased) / testloaderDiseased.batch_size)):
        img = data['image']
        test_images.append(img)

    return seg_images, test_images


test_images = list()
seg_images = list()

os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_011")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_011', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_013")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_013', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_072")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_072', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_089")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_089', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_127")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_127', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_148")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_148', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_189")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_189', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_237")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_237', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_238")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_238', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_248")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_248', 196, 196)
os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_273")
seg_images, test_images = get_real_anomaly_seg(seg_images, test_images, '../Mets_273', 196, 196)

print(len(seg_images))


def training_1_patient(seg_images, test_images):
    import torch
    import torch.nn as nn
    import torch.optim as opt
    import torchvision.transforms as transforms
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch.multiprocessing as mp

    """Reproducibility"""
    torch.manual_seed(0)

    """CUDA"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    SETTINGS
    """
    TRAIN = True
    LOAD_BEST_MODEL = True
    PARAMS_TO_LOAD = 9

    """
    PARAMETERS
    """
    # DATASET
    batch_size = 1
    img_width = 196
    img_height = 196

    # FACE GEN MODEL
    kernel_size_face_gen = 3
    init_channels_face_gen = 16
    stride_face_gen = 1
    padding_face_gen = 0

    # TRAINING
    epochs = 500
    patience_d = 20

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    # plant = 'pepper'
    brain = 'metshare'
    plantVillageTrain, plantVillageTestHealthy, plantVillageTestDiseased, plantVillageTestDiseasedMaskSeg = generateDataset(
        brain, transform)

    trainloader = get_training_dataloader(plantVillageTrain, batch_size)
    testloaderDiseased = get_test_dataloader(plantVillageTestDiseased, batch_size=1)
    testloaderHealthy = get_test_dataloader(plantVillageTestHealthy, batch_size=1)
    testloaderDiseasedMask = get_test_dataloader(plantVillageTestDiseasedMaskSeg, batch_size=1)

    training_set_per_shuffle = []
    for j, data in tqdm(enumerate(trainloader), total=int(len(plantVillageTrain) / trainloader.batch_size)):
        training_set_per_shuffle.append(data['image'])
    print(len(training_set_per_shuffle))

    # scommenta se vuoi griglia patches nere
    '''masked_data = []
    original_image = np.array(np.squeeze(training_set_per_shuffle[0]))
    maschera = np.zeros((img_width, img_width))
    # show_grey(original_image)
    count = np.nonzero(original_image)
    min_x = np.min(count[0])
    min_y = np.min(count[1])
    max_x = np.max(count[0])
    max_y = np.max(count[1])

    thirty_per_cent_x = int(((max_x - min_x) / 100) * 10)
    thirty_per_cent_y = int(((max_y - min_y) / 100) * 10)

    maschera[min_x:max_x, min_y:max_y] = 255


    for i in range(int((max_x-min_x) / thirty_per_cent_x)):
        start_x = min_x + i * thirty_per_cent_x
        for j in range(int((max_y - min_y) / thirty_per_cent_y)):
            or_copy = np.copy(original_image)
            start_y = min_y + j * thirty_per_cent_y
            or_copy[start_x: start_x + thirty_per_cent_x,start_y: start_y + thirty_per_cent_y] = 0
            or_copy = torch.from_numpy(or_copy)
            or_copy = torch.unsqueeze(or_copy,dim = 0)
            or_copy = torch.unsqueeze(or_copy,dim = 0)
            masked_data.append(or_copy)'''

    masked_data = []
    for i in range(len(training_set_per_shuffle)):
        masked_data.append([])

    for i in range(len(training_set_per_shuffle)):
        for j in range(5):
            masked_data[i].append(create_black_patch(np.array(np.squeeze(training_set_per_shuffle[i])), img_width))
        # masked_data.append(torch.unsqueeze(black_mask(training_set_per_shuffle[i],img_width), dim=0))

    print(np.shape(masked_data))

    '''for i in range(len(training_set_per_shuffle)):
        display(Image.fromarray(np.uint8(np.squeeze(training_set_per_shuffle[i])*255)))
        display(Image.fromarray(np.uint8(np.squeeze(masked_data[i])*255)))'''

    '''c = list(zip(training_set_per_shuffle, masked_data))
    random.shuffle(c)
    training_set_per_shuffle, masked_data = zip(*c)
    for i in range(len(training_set_per_shuffle)):
        display(Image.fromarray(np.uint8(np.squeeze(training_set_per_shuffle[i])*255)))
        display(Image.fromarray(np.uint8(np.squeeze(masked_data[i])*255)))'''

    """
    MODEL TRAINING
    """

    num_training_updates = 15000

    num_hiddens = 128
    num_residual_layers = 2
    num_residual_hiddens = 32
    num_embeddings = 64
    embedding_dim = 128
    commitment_cost = 0.30
    decay = 0
    # non utilizzato

    learning_rate = 1e-3

    # initialize the model
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay).to(device)

    criterion = nn.MSELoss(reduction='sum')

    optimizer = opt.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    # scheduler = ExponentialLR(optimizer, gamma=0.99)

    compute_loss = lambda a, b, c: a

    input_shape = (1, img_height, img_width)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=100, threshold=0.0001,
                                                           threshold_mode='abs', verbose=True)

    ddp_model = torch.nn.DataParallel(model)

    early_stopping = 500

    # questo funziona sul singolo testset del paziente
    '''engine = Engine(model=model,
                    trainloader=trainloader,
                    trainset=plantVillageTrain,
                    # testloader sarebbe il validation, tanto qui non lo usi 
                    testloader=trainloader,
                    testset=plantVillageTrain,
                    testloadermask = plantVillageTestDiseasedMaskSeg,
                    testsetmask = testloaderDiseasedMask,  
                    testloaderdiseased = testloaderDiseased,
                    testdiseasedset = plantVillageTestDiseased,
                    epochs=epochs,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler = scheduler,
                    input_shape=input_shape,
                    compute_loss=compute_loss,
                    device=device,
                    early_stopping= early_stopping,
                    patience_dice = patience_d)'''

    # così controlliamo tutti i test set durante il traininig del singolo modello
    engine = Engine(model=model,
                    trainloader=trainloader,
                    trainset=plantVillageTrain,
                    # testloader sarebbe il validation, tanto qui non lo usi
                    testloader=trainloader,
                    testset=plantVillageTrain,
                    testloadermask=plantVillageTestDiseasedMaskSeg,
                    testsetmask=seg_images,
                    testloaderdiseased=test_images,
                    testdiseasedset=plantVillageTestDiseased,
                    epochs=epochs,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    img_width=img_width,
                    img_height=img_height,
                    compute_loss=compute_loss,
                    device=device,
                    early_stopping=early_stopping,
                    patience_dice=patience_d,
                    training_set_per_shuffle=training_set_per_shuffle,
                    masked_data=masked_data)

    model, tot_loss, vq_loss, ce_loss, best_single_mean_distribution, best_single_std_distribution = engine.start1()

    for i in range(len(best_single_std_distribution)):
        print('Mean')
        print(best_single_mean_distribution[i])
        print('Std')
        print(best_single_std_distribution[i])

    print('Vq loss')
    plt.plot(vq_loss)
    plt.yscale("log")
    plt.show()

    '''print('Ce loss')
    plt.plot(ce_loss)
    plt.yscale("log")
    plt.show()'''

    print('Tot loss')
    plt.plot(tot_loss)
    plt.yscale("log")
    plt.show()

    model.eval()
    residual_volume = list()

    # mi calcolo tutti i residual e creo le due matrici di std e mean

    for j, data in tqdm(enumerate(trainloader), total=int(len(plantVillageTrain) / trainloader.batch_size)):
        img = data['image']
        img = img.cuda()
        reconstruction, loss, perplexity = model(img)

        for x in range(len(img)):
            rec = reconstruction[x, 0, :, :].cpu().detach().numpy()
            original = img[x, 0, :, :].cpu()
            diff = np.subtract(np.array(original), np.array(rec), dtype=np.float64)
            # diff = diff.clip(min=0)
            residual_volume.append(diff)

    pixels = list()
    std = np.zeros((img_height, img_width))
    mean = np.zeros((img_height, img_width))

    for i in tqdm(range(img_height)):
        for j in range(img_width):
            for index in range(len(residual_volume)):
                pixels.append(residual_volume[index][i][j])
                if len(pixels) == len(
                        residual_volume):  # se entro vuol dire ho fatto una lista dello stesso pixel di tutte le imm
                    numpy_pixels = np.array(pixels, dtype=np.float64)
                    std_0 = np.std(numpy_pixels, dtype=np.float64)
                    mean_0 = np.mean(numpy_pixels, dtype=np.float64)
                    std[i][j] = std_0
                    mean[i][j] = mean_0
                    pixels.clear()

    from matplotlib.pyplot import figure, imshow, axis
    from matplotlib.image import imread

    a = 0

    value = np.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5])
    k_collection = np.zeros(np.shape(value)[0])
    first_factor = np.zeros(np.shape(value)[0])
    second_factor = np.zeros(np.shape(value)[0])

    f_factor = []

    # Liste per salvarmi tutti i fattori dei parametri

    # Create List of empty list
    for i in range(len(value)):
        f_factor.append([])

    s_factor = []

    # Create List of empty list
    for i in range(len(value)):
        s_factor.append([])
    images = list()

    total_dice_score = 0

    for i in range(len(seg_images)):
        real_mask = seg_images[i][0, 0, :, :].cpu().detach().numpy()

        reconstruction, loss, perplexity = model(test_images[i].cuda())

    for i in tqdm(range(len(seg_images))):

        img = test_images[i].cuda()
        seg = seg_images[i].cuda()

        real_mask = seg[0, 0, :, :].cpu().detach().numpy()

        reconstruction, loss, perplexity = model(img)

        rec = np.array(reconstruction[0, 0, :, :].cpu().detach().numpy()) * 255
        rec = rec.clip(min=0)

        test_image = np.array(img[0, 0, :, :].cpu().detach().numpy()) * 255

        # print('Image number: ',i)

        diff = np.subtract(test_image, rec, dtype=np.float64)

        full_mask, resized_mask = create_images_masks(img[0, 0, :, :].cpu())

        count = 0
        for x in value:
            maps = find_anomalies(diff / 255, mean, std, x, img_width, img_height)
            mu = ((np.bitwise_and((resized_mask / 255).astype(int), (maps / 255).astype(int))) * 255).astype(int)
            dice_loss, a, b = new_dice(real_mask, mu / 255)
            k_collection[count] += dice_loss
            first_factor[count] += a
            second_factor[count] += b
            f_factor[count].append(a)
            s_factor[count].append(b)
            count += 1

    count = 0
    count_max = 0
    max_k = 0
    k = 0
    for x in value:

        if (k_collection[count] / len(seg_images)) > max_k:
            max_k = (k_collection[count] / len(seg_images))
            count_max = count
            k = x

        print('For value of k: ', x,
              ' the dice loss is: ', (k_collection[count] / len(seg_images)),
              ' with first factor: ', (first_factor[count] / len(seg_images)),
              ' with second factor: ', (second_factor[count] / len(seg_images)), '\n')

        count += 1

    print('For value of k: ', k, ' we have the maximum dice score of: ', max_k)

    anomaly_mask_images = list()
    anomaly_mask_images.clear()

    d = list()
    f = list()
    s = list()

    print('Lunghezza test set: ', len(seg_images))
    for i in tqdm(range(len(seg_images))):
        real_mask = seg_images[i][0, 0, :, :].cpu().detach().numpy()

        reconstruction, loss, perplexity = model(test_images[i].cuda())

        rec = np.array(reconstruction[0, 0, :, :].cpu().detach().numpy()) * 255

        test_image = np.array(test_images[i][0, 0, :, :].cpu().detach().numpy()) * 255

        diff = np.subtract(test_image, rec)

        full_mask, resized_mask = create_images_masks(test_images[i][0, 0, :, :].cpu())

        # print('Shape real mask: ',np.shape(real_mask),np.max(real_mask))

        images.clear()
        maps = find_anomalies(diff / 255, mean, std, k, img_width, img_height)
        mu = (resized_mask / 255) * maps
        dice_loss, a, b = new_dice(real_mask, mu / 255)
        # print('Shape real mask: ',np.shape(mu),np.max(mu))

        d.append(dice_loss)
        f.append(a)
        s.append(b)

        images.append(test_image)
        images.append(rec)
        images.append(diff)
        # images.append(real_mask)
        images.append(mu / 255)
        anomaly_mask_images.append((Image.fromarray(np.uint8(Image.fromarray(np.uint8(maps))))))

        print('Evaluated image with k = ', k)
        print('Dice : ', dice_loss, ' First factor: ', a, ' Second factor ', b)

        showImagesHorizontally(images)

    print('The general scores of the entire testsets are: ')

    print('Dice: ', np.sum(d) / len(seg_images), ' First factor: ', np.sum(f) / len(seg_images), ' second factor: ',
          np.sum(s) / len(seg_images))
    return model


'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_011")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_011")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_011")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_011")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_013")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_013")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_013")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_013")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_072")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_072")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_072")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_072")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_089")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_089")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_089")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_089")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_127")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_127")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_127")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_127")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_148")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_148")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_148")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_148")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_189")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_189")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_189")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_189")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_237")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_237")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_237")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_237")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_238")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_238")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_238")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_238")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_248")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_248")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_248")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_248")'
'''os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_273")
model = training_1_patient(seg_images,test_images)
torch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_273")'''
'os.chdir("/kaggle/input/11-paz-norm-centred-crop/Mets_273")\nmodel = training_1_patient(seg_images,test_images)\ntorch.save(model.state_dict(), f"/kaggle/working/outputs/best_param/best_model_273")'


###########################################################################################################################################
def ensemble_single_distribution():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dim = 196
    img_width = 196
    if not os.path.exists('/kaggle/working/all/ensemble_anomaly_masks'):
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks')
        os.mkdir('/kaggle/working/all/Anomaly_masks')
        os.mkdir('/kaggle/working/all/Filtered_ensemble')
        os.mkdir('/kaggle/working/all/Ensemble_mask')
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks_horizontal')
        os.mkdir('/kaggle/working/all/Anomaly_maps_horizontal')
        os.mkdir('/kaggle/working/all/Arrays')
        get_seg_and_test_images(seg_images, test_images, r'/kaggle/input/test-images/test/seg_paz_malati',
                                r'/kaggle/input/test-images/test/imm_paz_malati', image_dim)

    if len(seg_images) > 79:
        seg_images.clear()
        test_images.clear()
        get_seg_and_test_images(seg_images, test_images, r'/kaggle/input/test-images/test/seg_paz_malati',
                                r'/kaggle/input/test-images/test/imm_paz_malati', image_dim)
    print(len(seg_images))

    patients_name_list = get_patients_name_list(r'/kaggle/input/11-paz-norm-centred-crop')

    patients_image_folders_path = get_patients_images_path(r'/kaggle/input/11-paz-norm-centred-crop')

    patients_models_folders_path = get_patients_models_path(r'/kaggle/input/models-single-distribution',
                                                            patients_name_list)

    path_to_save_ensemble_images = r'/kaggle/working/all/ensemble_anomaly_masks'

    patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/working/all/Anomaly_masks', patients_name_list)

    save_filtered_ensemble_path = r'/kaggle/working/all/Filtered_ensemble'

    patients_anomaly_mask_horizontally_saving_path = create_saving_folders(
        r'/kaggle/working/all/Anomaly_maps_horizontal', patients_name_list)

    transform = transforms.Compose([
        transforms.Resize((img_width, img_width)),
        transforms.ToTensor()])

    num_hiddens = 128
    num_residual_layers = 2
    num_residual_hiddens = 32
    num_embeddings = 64
    embedding_dim = 128
    commitment_cost = 0.30
    decay = 0

    # non utilizzato
    # non utilizzato

    learning_rate = 1e-3
    # initialize the model
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay).to(device)

    ensemble_mask = np.zeros((image_dim, image_dim))

    for o in range(len(patients_models_folders_path)):
        # migliori tre ('Mets_013', 'Mets_072', 'Mets_237')
        # pat = patients_models_folders_path[o].split("_")[-1]
        # if pat == '013' or pat == '072' or pat == '237':

        # dizionario composto in questo modo: chiave paziente, valori media e std distribuzione dei
        # residui sui vai pazienti
        diz_patient_mean_std = {}

        print('Evaluating patient: ', patients_models_folders_path[o].split("_")[-1])

        model.load_state_dict(torch.load(patients_models_folders_path[o], map_location=torch.device('cpu')))
        model.eval()

        best_single_mean_distribution = []
        best_single_std_distribution = []
        single_mean_distribution = []
        single_std_distribution = []

        value = np.array(
            [1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.4, 3.5])
        value = np.array([2])

        total_dice_score = 0

        k_collection = np.zeros(np.shape(value)[0])
        first_factor = np.zeros(np.shape(value)[0])
        second_factor = np.zeros(np.shape(value)[0])

        f_factor = []

        # Liste per salvarmi tutti i fattori dei parametri
        for i in range(len(value)):
            f_factor.append([])

        s_factor = []

        # Create List of empty list
        for i in range(len(value)):
            s_factor.append([])

        dice_equal_to_zero = []

        # Create List of empty list
        for i in range(len(value)):
            dice_equal_to_zero.append(0)

        single_mean_distribution.clear()
        single_std_distribution.clear()
        count = 0

        for z in tqdm(range(len(patients_image_folders_path))):

            count_dice_uguale_a_zero = 0

            health = patients_image_folders_path[z] + '/brainmetshare/metshare/train/healthy/id/2'
            disease = patients_image_folders_path[z] + '/brainmetshare/metshare/test/disease/id/2'
            seg = patients_image_folders_path[z] + '/brainmetshare/metshare/test/disease/id/seg'

            # healthy
            h = sorted(glob.glob(health + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

            # disease
            d = sorted(glob.glob(disease + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

            # seg
            s = sorted(glob.glob(seg + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

            std_mask = np.zeros((img_width, img_width))
            count_batch = 0

            for img in h:
                im = Image.open(img)
                im = transform(im)
                std_mask += np.array(np.squeeze(im))
                im = torch.unsqueeze(im, 0)
                im = im.to(device)
                reconstruction, r_loss, perplexity = model(im)
                reconstruction = np.squeeze(reconstruction)
                reconstruction[reconstruction < 4 / 255] = 0

                original = np.squeeze(im)
                diff = np.subtract(original.cpu(), np.array(reconstruction.cpu().detach().numpy()))
                diff = np.array(diff).flatten()
                diff = diff[diff != 0]
                if count_batch == 0:
                    flat_residual_distribution = np.copy(diff)
                else:
                    flat_residual_distribution = np.concatenate((flat_residual_distribution, diff), axis=0)
                count_batch += 1
                # print('Len flat_residual_distribution', len(flat_residual_distribution))

            for img in d:
                im = Image.open(img)
                im = transform(im)
                im = torch.unsqueeze(im, 0)
                std_mask += np.array(np.squeeze(im))
                im = im.to(device)
                reconstruction, r_loss, perplexity = model(im)
                reconstruction[reconstruction < 4 / 255] = 0
                reconstruction = np.squeeze(reconstruction)
                original = np.squeeze(im)
                diff = np.subtract(original.cpu(), reconstruction.cpu().detach().numpy())
                diff = np.array(diff).flatten()
                diff = diff[diff != 0]
                flat_residual_distribution = np.concatenate((flat_residual_distribution, diff), axis=0)
                # print('Len flat_residual_distribution', len(flat_residual_distribution))

            single_mean = np.mean(flat_residual_distribution)
            single_std = np.std(flat_residual_distribution)

            diz_patient_mean_std[str(patients_image_folders_path[z].split("_")[-1])] = [single_mean, single_std]
            single_mean_distribution.append(single_mean)
            single_std_distribution.append(single_std)
            std_mask[std_mask > 0] = 1

            ensemble_mask += std_mask

            # calcolo dice di imm malate
            for y in range(len(d)):

                im = Image.open(d[y])
                im = transform(im)
                im = torch.unsqueeze(im, 0)
                im = im.to(device)

                se = Image.open(s[y])
                se = transform(se)
                se = np.squeeze(se)

                reconstruction, r_loss, perplexity = model(im)

                reconstruction = np.squeeze(reconstruction.cpu().detach().numpy())
                diff = np.subtract(np.squeeze(im.cpu().detach().numpy()) * 255, np.array(reconstruction) * 255)

                full_mask, resized_mask = create_images_masks(im[0, 0, :, :].cpu())
                # print('resize mask')

                # display(Image.fromarray(np.uint8(resized_mask)))
                count_coll = 0

                # for x in value:
                for x in range(len(value)):
                    maps = find_anomalies_single_value(diff / 255, single_mean, single_std, value[x], img_width)
                    mu = (resized_mask / 255) * (maps)
                    mu = std_mask * mu
                    dice_loss, a, b = new_dice(se, mu / 255)
                    if dice_loss == 0: dice_equal_to_zero[x] += 1
                    k_collection[x] += dice_loss
                    first_factor[x] += a
                    second_factor[x] += b
                    f_factor[x].append(a)
                    s_factor[x].append(b)

                count += 1

            max_k = 0
            k = 0
            count_coll = 0
            for x in value:

                if (k_collection[count_coll] / count) > max_k:
                    max_k = (k_collection[count_coll] / count)
                    k = x

                '''print('For value of k: ', x,
                      ' the dice loss is: ', (k_collection[count_coll] / count),
                      ' with first factor: ', (first_factor[count_coll] / count),
                      ' with second factor: ', (second_factor[count_coll] / count),
                      'missed anomalies',dice_equal_to_zero[count_coll],'n')'''
                count_coll += 1

        # scrivi codice per calocare dice con k migliore e per salvare anomaly maps

        blue_pixels_list = []
        blue_pixels_list.clear()

        anomaly_mask_images = list()
        anomaly_mask_images.clear()

        di = list()
        fi = list()
        si = list()
        images = []
        images_for_radius_cleaning = []
        images_for_radius_cleaning.clear()

        optimal_K = []

        count_image = 0

        print('Best k is: ', k)

        print(diz_patient_mean_std[str(patients_image_folders_path[z].split("_")[-1])][0],
              diz_patient_mean_std[str(patients_image_folders_path[z].split("_")[-1])][1])

        for p in tqdm(range(len(patients_image_folders_path))):

            disease = patients_image_folders_path[p] + '/brainmetshare/metshare/test/disease/id/2'
            seg = patients_image_folders_path[p] + '/brainmetshare/metshare/test/disease/id/seg'

            # disease
            d = sorted(glob.glob(disease + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

            # seg
            s = sorted(glob.glob(seg + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

            for v in range(len(d)):
                im = Image.open(d[v])
                im = transform(im)
                im = torch.unsqueeze(im, 0)

                im = im.to(device)

                se = Image.open(s[v])
                se = transform(se)
                se = np.squeeze(se)

                reconstruction, r_loss, perplexity = model(im)

                reconstruction = np.squeeze(reconstruction.cpu().detach().numpy())
                diff = np.subtract(np.squeeze(im.cpu().detach().numpy()) * 255, np.array(reconstruction) * 255)

                full_mask, resized_mask = create_images_masks(im[0, 0, :, :].cpu())

                maps = find_anomalies_single_value(diff / 255, diz_patient_mean_std[
                    str(patients_image_folders_path[z].split("_")[-1])][0],
                                                   diz_patient_mean_std[
                                                       str(patients_image_folders_path[z].split("_")[-1])][1], k,
                                                   img_width)
                mu = (resized_mask / 255) * (maps)

                intersection, blue_pixels = check_std(std=std_mask, anomaly_map=mu / 255)
                # Image.fromarray(np.uint8(blue_pixels * 255)).show()

                mu = intersection * 255

                # mu min,max 0,255

                dice_loss, a, b = new_dice(se, mu / 255)

                # print('Image number: ',count_image,'Dice: ', dice_loss, ' First factor: ', a,
                # second factor: ',b)

                optimal_K.append(optimal_k(diff / 255, se,
                                           diz_patient_mean_std[str(patients_image_folders_path[z].split("_")[-1])][0],
                                           diz_patient_mean_std[str(patients_image_folders_path[z].split("_")[-1])][1]))
                images_for_radius_cleaning.append(mu)

                di.append(dice_loss)
                fi.append(a)
                si.append(b)

                images.clear()
                images.append(np.squeeze(im.cpu().detach().numpy()) * 255)
                images.append(reconstruction * 255)
                images.append(diff)
                images.append(se * 255)
                images.append(mu / 255)
                images.append(blue_pixels)

                blue_pixels_list.append(blue_pixels)

                anomaly_mask_images.append((Image.fromarray(np.uint8(Image.fromarray(np.uint8(mu))))))

                # showImagesHorizontally(images,patients_anomaly_mask_horizontally_saving_path[o]
                #                   + str(count_image) + '.png')
                count_image += 1

        print('Optimal k for every image: ', optimal_K)
        print('Dice before cleaning')
        calculate_dice_anomalies_maps(images_for_radius_cleaning, seg_images)
        anomaly_mask_images.clear()
        anomaly_mask_images = clean_single_anomaly_single_pixels(images_for_radius_cleaning, image_dim)
        print('Dice after cleaning')
        calculate_dice_anomalies_maps(anomaly_mask_images, seg_images)

        # devo trasformare lista in imm PIL
        new_list_anomaly_maps = []
        for i in range(len(anomaly_mask_images)):
            new_list_anomaly_maps.append(Image.fromarray(np.uint8(Image.fromarray(np.uint8(anomaly_mask_images[i])))))

        for i in range(len(new_list_anomaly_maps)):
            three_channels_AM(np.array(new_list_anomaly_maps[i]) / 255, np.array(np.squeeze(seg_images[i])),
                              blue_pixels_list[i]).save(patients_anomaly_mask_saving_path[o] + str(i) + '.png')

    ensemble_mask[ensemble_mask < len(patients_image_folders_path)] = 0
    ensemble_mask[ensemble_mask >= len(patients_image_folders_path)] = 1

    if len(seg_images) > 79:
        seg_images.clear()
        test_images.clear()
        get_seg_and_test_images(seg_images, test_images, r'/kaggle/input/test-images/test/seg_paz_malati',
                                r'/kaggle/input/test-images/test/imm_paz_malati', image_dim)
    print(len(seg_images))

    number_of_models = len(patients_models_folders_path)
    number_of_images = len(seg_images)

    print(number_of_images)

    # in gray scale
    anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
        number_of_models,
        patients_anomaly_mask_saving_path)

    # qui sono a colori ricorda
    coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images, image_dim,
                                                      number_of_models,
                                                      True, path_to_save_ensemble_images, seg_images, ensemble_mask,
                                                      image_dim, test_images)
    d = []
    f = []
    s = []

    for i in range(len(coloured_anomaly_maps)):
        dice_loss, a, b = new_dice(np.array(seg_images[i][0, 0, :, :]),
                                   np.array(coloured_anomaly_maps[i])[:, :, 1] / 255)
        recall, precision, f1_score, FNR, FPR = define_metrics(np.array(seg_images[i][0, 0, :, :]),
                                                               np.array(coloured_anomaly_maps[i])[:, :, 1] / 255)
        print('Dice : ' + str(dice_loss) + ' First factor: ' + str(a) + ' Second factor ' + str(b) +
              'Recall: ' + str(recall) + 'Precision: ' + str(precision) + 'F1_score: ' + str(f1_score) +
              'FNR: ' + str(FNR) + 'FPR: ' + str(FPR))

        d.append(dice_loss)
        f.append(a)
        s.append(b)

    print('\nThe general scores of the entire testsets are: ')

    print('Dice: ', np.sum(d) / len(seg_images), ' First factor: ', np.sum(f) / len(seg_images), ' second factor: ',
          np.sum(s) / len(seg_images))

    ensemble_anomaly_maps = create_list_with_anomaly_mask_from_ensemble(path_to_save_ensemble_images)

    list_anomalies_maps = eliminate_radius_in_ensemble_anomaly_map(ensemble_anomaly_maps, save_filtered_ensemble_path,
                                                                   test_images, seg_images, True)

    # create_3X3_anomalies_map(list_anomalies_maps,seg_images)'''


def iterative_combinations_ensemble(best_3, input_path_anomaly_maps, patients_name_list, image_dim, FALSE, seg_images,
                                    test_images):
    # questa funzione la chiami più volte
    best_dice = 0
    best_first = 0
    best_second = 0
    best_3_copy = best_3.copy()
    '''for patient in patients_name_list:
        best_3_copy = best_3.copy()'''

    '''if patient not in best_3:
            best_3_copy.append(patient)'''
    # print(best_3 , selected_patients)
    dice_list.clear()
    first_list.clear()
    second_list.clear()
    print(best_3_copy)
    number_of_models = len(best_3_copy)
    patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/working/all/Anomaly_masks', best_3_copy)
    patients_anomaly_mask_saving_path = create_saving_folders(input_path_anomaly_maps, best_3_copy)

    ensemble_mask = std_from_choosed_saved_arrays(r'/kaggle/input/arrays', best_3_copy)

    path_to_save_ensemble_images = r'/kaggle/working/all/ensemble_anomaly_masks'

    save_filtered_ensemble_path = r'/kaggle/working/all/Filtered_ensemble'

    ensemble_mask[ensemble_mask < len(best_3_copy)] = 0
    ensemble_mask[ensemble_mask >= len(best_3_copy)] = 1

    number_of_models = len(best_3_copy)
    number_of_images = len(test_images)

    # in gray scale
    anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
        number_of_models,
        patients_anomaly_mask_saving_path)

    # qui sono a colori ricorda
    coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images,
                                                      image_dim,
                                                      number_of_models,
                                                      True, path_to_save_ensemble_images,
                                                      seg_images,
                                                      ensemble_mask, image_dim, test_images)

    for m in range(len(coloured_anomaly_maps)):
        coloured_anomaly_maps[m] = np.array(coloured_anomaly_maps[m])
    # coloured_anomaly_maps[np.array(coloured_anomaly_maps)] = coloured_anomaly_maps

    dice, first, second = eliminate_radius_in_ensemble_anomaly_map(coloured_anomaly_maps,
                                                                   save_filtered_ensemble_path,
                                                                   test_images, seg_images, True)
    if dice > best_dice:
        best_dice = dice
        best_first = first
        best_second = second
        best_4 = best_3_copy.copy()

    print('Dice: ', dice, ' First factor: ', first, ' second factor: ', second)

    return best_4, best_dice, best_first, best_second


def choose_best_combination_of_ensemble_single_distribuion():
    seg_images = []
    test_images = []
    image_dim = 196

    if not os.path.exists('/kaggle/working/all/ensemble_anomaly_masks'):
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks')
        os.mkdir('/kaggle/working/all/Anomaly_masks')
        os.mkdir('/kaggle/working/all/Filtered_ensemble')
        os.mkdir('/kaggle/working/all/Ensemble_mask')
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks_horizontal')
        os.mkdir('/kaggle/working/all/Anomaly_maps_horizontal')
        os.mkdir('/kaggle/working/all/Arrays')

    # contains the name of patients use for the ensemble
    patients_name_list = get_patients_name_list(r'/kaggle/input/11-paz-norm-centred-crop')

    patients_image_folders_path = get_patients_images_path(r'/kaggle/input/11-paz-norm-centred-crop')

    patients_models_folders_path = get_choosed_patients_models_path(
        r'/kaggle/input/models-single-distribution/best_model_011',
        patients_name_list)

    # create list containing image test of all pateints
    seg_images = []
    test_images = []
    # sono le massime dimensioni di slice di ogni paziente min,max 0,1
    # max_slice_all_patients = []

    list_combinations = list(combinations(patients_name_list, 3))
    # codice per creare combinazione di 3 pazienti
    # np.save(r'C:\Users\paoli\Desktop\3_combinations' + r'.npy', list_combinations)

    print(len(list_combinations))
    count_best_patients_index = 0
    max_dice = 0
    # list_combinations = np.load(r'C:\Users\paoli\Desktop\3_combinations' + r'.npy', allow_pickle=True)
    if 0:
        for h in range(len(list_combinations)):
            selected_patients = list_combinations[h]
            dice_list.clear()
            first_list.clear()
            second_list.clear()

            patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/working/all/Anomaly_masks',
                                                                      selected_patients)

            patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/input/anomaly-maps-17-k',
                                                                      selected_patients)

            ensemble_mask = std_from_choosed_saved_arrays(r'/kaggle/input/arrays', selected_patients)

            path_to_save_ensemble_images = r'/kaggle/working/all/ensemble_anomaly_masks'

            save_filtered_ensemble_path = r'/kaggle/working/all/Filtered_ensemble'

            ensemble_mask[ensemble_mask < len(selected_patients)] = 0
            ensemble_mask[ensemble_mask >= len(selected_patients)] = 1

            # display(Image.fromarray(np.uint8(np.squeeze(ensemble_mask)*255)))

            number_of_models = len(selected_patients)
            number_of_images = len(test_images)

            # print(patients_anomaly_mask_saving_path)

            # in gray scale
            anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
                number_of_models,
                patients_anomaly_mask_saving_path)

            # print(np.shape(anomaly_mask_from_all_models))

            # qui sono a colori ricorda
            coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images, image_dim,
                                                              number_of_models,
                                                              True, path_to_save_ensemble_images, seg_images,
                                                              ensemble_mask, image_dim, test_images)

            for m in range(len(coloured_anomaly_maps)):
                coloured_anomaly_maps[m] = np.array(coloured_anomaly_maps[m])
            # coloured_anomaly_maps[np.array(coloured_anomaly_maps)] = coloured_anomaly_maps

            dice, first, second = eliminate_radius_in_ensemble_anomaly_map(coloured_anomaly_maps,
                                                                           save_filtered_ensemble_path,
                                                                           test_images, seg_images, True)

            # print('Dice: ', dice, ' First factor: ', first, ' second factor: ', second)
            if dice > max_dice:
                max_dice = dice
                count_best_patients_index = h

        print('Best combination of 3 patients: ', list_combinations[count_best_patients_index])

    # best_combination = list(list_combinations[count_best_patients_index]).copy()
    best_combination = ['Mets_013', 'Mets_072', 'Mets_237']

    for i in range(len(patients_name_list) - 3):
        best_combination, best_d, best_f, best_s = iterative_combinations_ensemble(best_combination.copy(),
                                                                                   patients_name_list, image_dim, False,
                                                                                   seg_images,
                                                                                   test_images)
        print(best_combination)
        # print('Dice: ', best_d, ' First factor: ', best_f, ' second factor: ', best_s)


def choose_best_combination_of_ensemble_single_distribuion_3_patients_16_18_2():
    image_dim = 196
    seg_images = []
    test_images = []
    get_seg_and_test_images(seg_images, test_images, r'/kaggle/input/test-images/test/seg_paz_malati',
                            r'/kaggle/input/test-images/test/imm_paz_malati', image_dim)

    if not os.path.exists('/kaggle/working/all/ensemble_anomaly_masks'):
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks')
        os.mkdir('/kaggle/working/all/Anomaly_masks')
        os.mkdir('/kaggle/working/all/Filtered_ensemble')
        os.mkdir('/kaggle/working/all/Ensemble_mask')
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks_horizontal')
        os.mkdir('/kaggle/working/all/Anomaly_maps_horizontal')
        os.mkdir('/kaggle/working/all/Arrays')

    # contains the name of patients use for the ensemble
    patients_name_list = get_patients_name_list(r'/kaggle/input/best-2-anom-maps-16-18-2')
    print(patients_name_list)

    patients_image_folders_path = get_patients_images_path(r'/kaggle/input/11-paz-norm-centred-crop')

    '''patients_models_folders_path = get_choosed_patients_models_path(r'/kaggle/input/matrix-models',
                                                                    patients_name_list)'''

    # create list containing image test of all pateints
    # sono le massime dimensioni di slice di ogni paziente min,max 0,1
    # max_slice_all_patients = []

    list_combinations = list(combinations(patients_name_list, 2))
    # codice per creare combinazione di 3 pazienti
    # np.save(r'C:\Users\paoli\Desktop\3_combinations' + r'.npy', list_combinations)

    index_to_delete = []
    new_list_combinations = []
    for g in range(len(list_combinations)):
        if list_combinations[g][0].split('_')[1] == list_combinations[g][1].split('_')[1]: index_to_delete.append(g)

    for g in range(len(list_combinations)):
        if g not in index_to_delete: new_list_combinations.append(list_combinations[g])

    count_best_patients_index = 0
    max_dice = 0
    # list_combinations = np.load(r'C:\Users\paoli\Desktop\3_combinations' + r'.npy', allow_pickle=True)

    '''for h in range(len(new_list_combinations)):
        selected_patients = new_list_combinations[h]
        print(selected_patients)
        dice_list.clear()
        first_list.clear()
        second_list.clear()

        patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/working/all/Anomaly_masks',selected_patients)

        #DEVI METTERE NUOVE ANOMALY_MAPS
        patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/input/best-2-anom-maps-16-18-2',selected_patients)

        ensemble_mask = std_from_choosed_saved_arrays(r'/kaggle/input/arrays', selected_patients)

        path_to_save_ensemble_images = r'/kaggle/working/all/ensemble_anomaly_masks'

        save_filtered_ensemble_path = r'/kaggle/working/all/Filtered_ensemble'

        ensemble_mask[ensemble_mask < len(selected_patients)] = 0
        ensemble_mask[ensemble_mask >= len(selected_patients)] = 1

        #display(Image.fromarray(np.uint8(np.squeeze(ensemble_mask)*255)))


        number_of_models = len(selected_patients)
        number_of_images = len(test_images)


        #print(patients_anomaly_mask_saving_path)


        # in gray scale
        anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
            number_of_models,
            patients_anomaly_mask_saving_path)

        # qui sono a colori ricorda
        coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images, image_dim,
                                                          number_of_models,
                                                          True, path_to_save_ensemble_images, seg_images,
                                                          ensemble_mask, image_dim, test_images)

        for m in range(len(coloured_anomaly_maps)):
            coloured_anomaly_maps[m] = np.array(coloured_anomaly_maps[m])
        # coloured_anomaly_maps[np.array(coloured_anomaly_maps)] = coloured_anomaly_maps



        dice, first, second = eliminate_radius_in_ensemble_anomaly_map(coloured_anomaly_maps,
                                                                       save_filtered_ensemble_path,
                                                                       test_images, seg_images,True)

        #print('Dice: ', dice, ' First factor: ', first, ' second factor: ', second)
        if dice>max_dice:
            max_dice = dice
            count_best_patients_index = h

        print('Best combination of ',len(patients) ,' patients: ', list_combinations[count_best_patients_index])

    best_combination = list(list_combinations[count_best_patients_index]).copy()
    last_patient_to_try = []
    for pat in patients_name_list:
        if pat not in best_combination and best_combination[0].split('_')[1] != pat.split('_')[1] and best_combination[1].split('_')[1] != pat.split('_')[1]:
            last_patient_to_try.append(pat)

    #('Mets_072_2', 'Mets_013_16') 
    for pat in last_patient_to_try:
        selected_patients = best_combination.copy()
        selected_patients.append(pat)
        print(selected_patients)
        best_combination,best_d,best_f,best_s = iterative_combinations_ensemble(selected_patients, patients_name_list, image_dim, False, seg_images,
                                            test_images)'''

    selected_patients = ['Mets_072_2', 'Mets_013_16', 'Mets_237_16']

    print(selected_patients)
    best_combination, best_d, best_f, best_s = iterative_combinations_ensemble(selected_patients, patients_name_list,
                                                                               image_dim, False, seg_images,
                                                                               test_images)

    selected_patients = ['Mets_072_2', 'Mets_013_16', 'Mets_237_18']

    print(selected_patients)
    best_combination, best_d, best_f, best_s = iterative_combinations_ensemble(selected_patients, patients_name_list,
                                                                               image_dim, False, seg_images,
                                                                               test_images)

    selected_patients = ['Mets_072_2', 'Mets_013_16', 'Mets_237_2']

    print(selected_patients)
    best_combination, best_d, best_f, best_s = iterative_combinations_ensemble(selected_patients, patients_name_list,
                                                                               image_dim, False, seg_images,
                                                                               test_images)

    # print('Dice: ', best_d, ' First factor: ', best_f, ' second factor: ', best_s)


def choose_best_combination_of_ensemble_matrix_distribuion():
    image_dim = 196
    seg_images = []
    test_images = []
    get_seg_and_test_images(seg_images, test_images, r'/kaggle/input/test-images/test/seg_paz_malati',
                            r'/kaggle/input/test-images/test/imm_paz_malati', image_dim)

    if not os.path.exists('/kaggle/working/all/ensemble_anomaly_masks'):
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks')
        os.mkdir('/kaggle/working/all/Anomaly_masks')
        os.mkdir('/kaggle/working/all/Filtered_ensemble')
        os.mkdir('/kaggle/working/all/Ensemble_mask')
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks_horizontal')
        os.mkdir('/kaggle/working/all/Anomaly_maps_horizontal')
        os.mkdir('/kaggle/working/all/Arrays')

    # contains the name of patients use for the ensemble
    patients_name_list = get_patients_name_list(r'/kaggle/input/11-paz-norm-centred-crop')

    patients_image_folders_path = get_patients_images_path(r'/kaggle/input/11-paz-norm-centred-crop')

    patients_models_folders_path = get_choosed_patients_models_path(r'/kaggle/input/matrix-models',
                                                                    patients_name_list)

    # create list containing image test of all pateints
    # sono le massime dimensioni di slice di ogni paziente min,max 0,1
    # max_slice_all_patients = []

    list_combinations = list(combinations(patients_name_list, 3))
    # codice per creare combinazione di 3 pazienti
    # np.save(r'C:\Users\paoli\Desktop\3_combinations' + r'.npy', list_combinations)

    print(len(list_combinations))
    count_best_patients_index = 0
    max_dice = 0
    # list_combinations = np.load(r'C:\Users\paoli\Desktop\3_combinations' + r'.npy', allow_pickle=True)

    for h in range(len(list_combinations)):
        selected_patients = list_combinations[h]
        print(selected_patients)
        dice_list.clear()
        first_list.clear()
        second_list.clear()

        patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/working/all/Anomaly_masks',
                                                                  selected_patients)
        # DEVI METTERE NUOVE ANOMALY_MAPS
        patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/input/anomaly-masks-matrix-patient',
                                                                  selected_patients)

        ensemble_mask = std_from_choosed_saved_arrays(r'/kaggle/input/arrays', selected_patients)

        path_to_save_ensemble_images = r'/kaggle/working/all/ensemble_anomaly_masks'

        save_filtered_ensemble_path = r'/kaggle/working/all/Filtered_ensemble'

        ensemble_mask[ensemble_mask < len(selected_patients)] = 0
        ensemble_mask[ensemble_mask >= len(selected_patients)] = 1

        # display(Image.fromarray(np.uint8(np.squeeze(ensemble_mask)*255)))

        number_of_models = len(selected_patients)
        number_of_images = len(test_images)

        # print(patients_anomaly_mask_saving_path)

        # in gray scale
        anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
            number_of_models,
            patients_anomaly_mask_saving_path)

        # print(np.shape(anomaly_mask_from_all_models))

        # qui sono a colori ricorda
        coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images, image_dim,
                                                          number_of_models,
                                                          True, path_to_save_ensemble_images, seg_images,
                                                          ensemble_mask, image_dim, test_images)

        for m in range(len(coloured_anomaly_maps)):
            coloured_anomaly_maps[m] = np.array(coloured_anomaly_maps[m])
        # coloured_anomaly_maps[np.array(coloured_anomaly_maps)] = coloured_anomaly_maps

        dice, first, second = eliminate_radius_in_ensemble_anomaly_map(coloured_anomaly_maps,
                                                                       save_filtered_ensemble_path,
                                                                       test_images, seg_images, True)

        # print('Dice: ', dice, ' First factor: ', first, ' second factor: ', second)
        if dice > max_dice:
            max_dice = dice
            count_best_patients_index = h

        print('Best combination of 3 patients: ', list_combinations[count_best_patients_index])

    best_combination = list(list_combinations[count_best_patients_index]).copy()
    # best_combination = ['Mets_013', 'Mets_072', 'Mets_237']

    for i in range(len(patients_name_list) - 3):
        best_combination, best_d, best_f, best_s = iterative_combinations_ensemble(best_combination.copy(),
                                                                                   patients_name_list, image_dim, False,
                                                                                   seg_images,
                                                                                   test_images)
        print(best_combination)
        # print('Dice: ', best_d, ' First factor: ', best_f, ' second factor: ', best_s)


def fine_testo_i_3_autoencoders_migliori_su_tutto_il_dataset():
    # lo uso per testare i 3 modelli su tutto il dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dim = 196
    img_width = 196
    if not os.path.exists('/kaggle/working/all/Anomaly_masks'):
        os.mkdir('/kaggle/working/all/Seg')
        os.mkdir('/kaggle/working/all/Orig')
        os.mkdir('/kaggle/working/all/Anomaly_masks')
        os.mkdir('/kaggle/working/all/Anomaly_masks_radius')

        get_seg_and_test_images(seg_images, test_images, r'/kaggle/input/test-images/test/seg_paz_malati',
                                r'/kaggle/input/test-images/test/imm_paz_malati', image_dim)

    patients_name_list = get_patients_name_list(r'/kaggle/input/tesi-test-set/Dataset_Finale_finale_patch')

    patients_image_folders_path = get_patients_images_path(r'/kaggle/input/tesi-test-set/Dataset_Finale_finale_patch')

    patients_models_folders_path = get_patients_models_path(r'/kaggle/input/models-single-distribution',
                                                            patients_name_list)

    path_to_save_ensemble_images = r'/kaggle/working/all/ensemble_anomaly_masks'

    patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/working/all/Anomaly_masks', patients_name_list)

    patients_anomaly_mask_saving_path = create_saving_folders(r'/kaggle/working/all/Anomaly_masks_r',
                                                              patients_name_list)

    save_filtered_ensemble_path = r'/kaggle/working/all/Filtered_ensemble'

    # patients_anomaly_mask_horizontally_saving_path = create_saving_folders(r'/kaggle/working/all/Anomaly_maps_horizontal',patients_name_list)

    transform = transforms.Compose([
        transforms.Resize((img_width, img_width)),
        transforms.ToTensor()])

    num_hiddens = 128
    num_residual_layers = 2
    num_residual_hiddens = 32
    num_embeddings = 64
    embedding_dim = 128
    commitment_cost = 0.30
    decay = 0

    # non utilizzato
    # non utilizzato

    learning_rate = 1e-3
    # initialize the model
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim, commitment_cost, decay).to(device)

    for o in range(len(patients_models_folders_path)):

        # ['Mets_072_2', 'Mets_013_16', 'Mets_237_18']

        # migliori tre ('Mets_013', 'Mets_072', 'Mets_237')
        pat = patients_models_folders_path[o].split("_")[-1]
        if pat == '013' or pat == '072' or pat == '237':

            if pat == '013':
                value = 1.6
                single_mean = -0.004889584
                single_std = 0.045074686

            elif pat == '072':
                value = 2
                single_mean = -0.0004645253
                single_std = 0.04490671

            elif pat == '237':
                value = 1.8
                single_mean = 0.00014915108
                single_std = 0.05165132

            std_mask = np.zeros((196, 196))
            std_mask += np.load(r'/kaggle/input/arrays' + '/' + 'Mets_' + pat + '.npy')
            # print(np.max(std_mask))

            # display(Image.fromarray(np.uint8(Image.fromarray(np.uint8(std_mask)*255))))

            print('Evaluating patient: ', patients_models_folders_path[o].split("_")[-1])

            model.load_state_dict(torch.load(patients_models_folders_path[o], map_location=torch.device('cpu')))
            model.eval()

            blue_pixels_list = []
            blue_pixels_list.clear()

            anomaly_mask_images = list()
            anomaly_mask_images.clear()

            images_for_radius_cleaning = []
            images_for_radius_cleaning.clear()
            seg_list = []

            original_image_to_save = list()

            count_image = 0

            for p in tqdm(range(len(patients_image_folders_path))):

                disease = patients_image_folders_path[p] + '/brainmetshare/metshare/test/disease/id/2'
                seg = patients_image_folders_path[p] + '/brainmetshare/metshare/test/disease/id/seg'

                # disease
                d = sorted(glob.glob(disease + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))
                # seg
                s = sorted(glob.glob(seg + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

                for v in range(len(d)):
                    im = Image.open(d[v])
                    original_image_to_save.append(im)
                    im = transform(im)
                    im = torch.unsqueeze(im, 0)
                    im = im.to(device)
                    se = Image.open(s[v])
                    se = transform(se)
                    se = np.squeeze(se)

                    reconstruction, r_loss, perplexity = model(im)

                    reconstruction = np.squeeze(reconstruction.cpu().detach().numpy())
                    diff = np.subtract(np.squeeze(im.cpu().detach().numpy()) * 255, np.array(reconstruction) * 255)

                    # full_mask, resized_mask = create_images_masks(im[0, 0, :, :].cpu().numpy()*255)

                    maps = find_anomalies_single_value(diff / 255, single_mean, single_std, value, img_width)

                    # se non aggiusto resized map
                    mu = maps

                    # senno
                    # mu = (resized_mask/255) * (maps)

                    intersection, blue_pixels = check_std(std=std_mask, anomaly_map=mu / 255)
                    # display(Image.fromarray(np.uint8(blue_pixels * 255)))

                    mu = intersection * 255

                    # mu min,max 0,255
                    blue_pixels_list.append(blue_pixels)
                    anomaly_mask_images.append(mu)
                    seg_list.append(se.detach().numpy())

                    count_image += 1

            print('Dice before cleaning')
            calculate_dice_anomalies_maps(anomaly_mask_images, seg_list)
            images_for_radius_cleaning.clear()
            images_for_radius_cleaning = clean_single_anomaly_single_pixels(anomaly_mask_images, image_dim)

            print('Dice after cleaning')
            calculate_dice_anomalies_maps(images_for_radius_cleaning, seg_list)

            for i in range(len(anomaly_mask_images)):
                three_channels_AM(anomaly_mask_images[i] / 255, seg_list[i], blue_pixels_list[i]).save(
                    r'/kaggle/working/all/Anomaly_masks/Mets_' + str(pat) + '/' + str(i) + '.png')
                Image.fromarray(np.uint8(seg_list[i] * 255)).save(r'/kaggle/working/all/Seg' + '/' + str(i) + '.png')
                original_image_to_save[i].save('/kaggle/working/all/Orig' + '/' + str(i) + '.png')
                three_channels_AM(images_for_radius_cleaning[i] / 255, seg_list[i], blue_pixels_list[i]).save(
                    r'/kaggle/working/all/Anomaly_masks_radius/Mets_' + str(pat) + '/' + str(i) + '.png')


def get_seg(seg_images, path_seg, img_width):
    transform = transforms.Compose([
        transforms.Resize((img_width, img_width)),
        transforms.ToTensor()])

    seg = sorted(glob.glob(path_seg + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

    count_seg = 0
    for img in seg:
        image = Image.open(img)
        image = np.array(image)
        image[image > 0] = 255
        image = transform(Image.fromarray(image))
        image = torch.unsqueeze(image, dim=0)
        seg_images.append(image)

    return seg_images


def best_final_combination_of_ensemble_single_distribuion_3_patients():
    image_dim = 196
    seg_images = []
    test_images = []

    get_seg_and_test_images(seg_images, test_images, r'/kaggle/input/dati-per-test-finale/Risultati finali/Seg',
                            r'/kaggle/input/dati-per-test-finale/Risultati finali/Orig', image_dim)

    if not os.path.exists('/kaggle/working/all/ensemble_anomaly_masks'):
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks')
        os.mkdir('/kaggle/working/all/Anomaly_masks')
        os.mkdir('/kaggle/working/all/Filtered_ensemble')
        os.mkdir('/kaggle/working/all/Ensemble_mask')
        os.mkdir('/kaggle/working/all/ensemble_anomaly_masks_horizontal')
        os.mkdir('/kaggle/working/all/Anomaly_maps_horizontal')
        os.mkdir('/kaggle/working/all/Arrays')

    # contains the name of patients use for the ensemble
    patients_name_list = get_patients_name_list(
        r'/kaggle/input/dati-per-test-finale/Risultati finali/Anomaly_masks_radius')

    print(patients_name_list)

    selected_patients = patients_name_list

    number_of_models = len(selected_patients)

    patients_anomaly_mask_saving_path = create_saving_folders(
        r'/kaggle/input/dati-per-test-finale/Risultati finali/Anomaly_masks_radius', selected_patients)

    ensemble_mask = std_from_choosed_saved_arrays(r'/kaggle/input/arrays', selected_patients)

    path_to_save_ensemble_images = r'/kaggle/working/all/ensemble_anomaly_masks'

    save_filtered_ensemble_path = r'/kaggle/working/all/Filtered_ensemble'

    ensemble_mask[ensemble_mask < len(selected_patients)] = 0
    ensemble_mask[ensemble_mask >= len(selected_patients)] = 1

    number_of_models = len(selected_patients)
    number_of_images = len(seg_images)

    print(number_of_images)

    # in gray scale
    anomaly_mask_from_all_models, list_to_know_the_order_of_patient_in_the_ensemble = create_list_with_anomaly_mask_from_all_models(
        number_of_models,
        patients_anomaly_mask_saving_path)

    # qui sono a colori ricorda
    coloured_anomaly_maps = get_ensemble_anomaly_maps(anomaly_mask_from_all_models, number_of_images,
                                                      image_dim,
                                                      number_of_models,
                                                      True, path_to_save_ensemble_images,
                                                      seg_images,
                                                      ensemble_mask, image_dim, test_images)

    for m in range(len(coloured_anomaly_maps)):
        coloured_anomaly_maps[m] = np.array(coloured_anomaly_maps[m])
    # coloured_anomaly_maps[np.array(coloured_anomaly_maps)] = coloured_anomaly_maps

    dice, first, second = eliminate_radius_in_ensemble_anomaly_map(coloured_anomaly_maps,
                                                                   save_filtered_ensemble_path,
                                                                   test_images, seg_images, True)


########################################################################################
# Autoencoder utilizzato
# VQVAEModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAEModel(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(VQVAEModel, self).__init__()

        # first number is the number of channel

        self._encoder = Encoder(1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                       commitment_cost)

        self._decoder = Decoder(embedding_dim,
                                num_hiddens,  # dim input decoder
                                num_residual_layers,
                                num_residual_hiddens)
        self.initialize_weights()

    def forward(self, x):

        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return x_recon, loss, perplexity

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                # occhio
                                                out_channels=1,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = nn.Dropout(0)(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        # in_channels = 1

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,  # è una divisione senza parte frazionaria
                                 kernel_size=4,
                                 stride=2, padding=1)

        # shape [batch,
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = nn.Dropout(0)(x)
        x = self._conv_3(x)
        return self._residual_stack(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

