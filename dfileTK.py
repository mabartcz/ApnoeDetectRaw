# dfileTK.py
# Tool kit for work with EASYS2 d-file
#
# Library with support functions for loading, preprocessing data in *.d format from Brainscope EASYS2
#
# Author:   Bc. Martin Barton
# Contact:  ma.barton@seznam.cz
# Date:     2020-18-05

import sys
import struct
import scipy
import numpy as np
from scipy import stats
import scipy.signal as sig
import scipy.fftpack
from window_slider import Slider
import collections

from matplotlib import pyplot as plt

from pytictoc import TicToc
t = TicToc()


def find(offset, length, mode, file):
    # Funkce pro vycteni dat ze souboru "file" v delce "length"
    # s adresací od začátku "offset" podle typu proměné "mode" viz struct fce
    file.seek(offset, 0)
    data = file.read(length)
    value = struct.unpack(mode, data)
    return value[0]

def save(offset, mode, data, file):
    # Funkce na prepsani dat- offset - kam to vlozi,
    # mode - jakej datovej typ, data - co vlozit
    file.seek(offset, 0)
    value = struct.pack(mode, data)
    file.write(value)

class DFile:
    # Main class for *.d file processing

    def __init__(self, main_file):

        self.main_file = main_file
        # Nacteni zakladnich dat
        file = open(main_file, "rb")
        self.file_name = main_file
        self.ftype = find(15, 1, '<B', file)  # Najde R nebo F type
        self.nchan = find(16, 1, '<b', file)  # Najde pocet kanalu
        self.fsamp = find(18, 2, '<H', file)  # Najde vzorkovaci frekvenci
        self.nsamp = find(20, 4, '<L', file)  # Najde celkovy pocet vzorku
        self.d_val = find(24, 1, '<B', file)  # Najde d_val
        self.unit  = find(25, 1, '<B', file)  # Najde prepocet
        self.zero  = find(26, 2, '<H', file)  # Najde offset dat
        self.data_org   = find(28, 2, '<H', file) * 16  # Najde na jakem bytu zacinaji data
        xhdr_org   = find(30, 2, '<H', file) * 16  # Najde zacatek extended headru

        # -----------------------------------------------------------------------
        # Kontrola jestli je nacten D file
        try:
            if chr(self.ftype) != "D":
                pass
        except:
            print(main_file)
            sys.exit("Not a D file !")

        # kontrola jestli je spravne kalibrovano
        # self.d_normal = '{0:08b}'.format(self.d_val)
        # self.d_normal = list(map(int, str(self.d_normal)))


        # -----------------------------------------------------------------------
        # Nacteni extendet headru do listu
        self.xhdr_ID = []  # nazec ext hlavicky v hexu
        self.xhdr_size = []  # velikost dat za hlavickou
        self.xhdr_index = []  # zacatek konkretniho xhdr
        f = xhdr_org
        counter = 1
        counter2 = 0
        h = 0
        while True:
            x = find(xhdr_org + h, 2, '<h', file)  # nacita ID extheadru
            g = find(xhdr_org + h + 2, 2, '<h', file)  # nacita delku extheadru
            h = h + g + 4
            if counter == 1:
                f = f + 4
                counter = 0
            else:
                f = f + self.xhdr_size[counter2 - 1] + 4
            if x == 0:
                break
            self.xhdr_ID.append(hex(x))
            self.xhdr_size.append(g)
            self.xhdr_index.append(f)
            counter2 += 1

        if "0x5454" in self.xhdr_ID:
            # nacteni velikosti a offsetu Table tags
            tt_index = self.xhdr_ID.index("0x5454")
            tt_def_len = find(self.xhdr_index[tt_index], 2, '<h', file)  # nacita delku definice tt
            self.tt_list_len = find(self.xhdr_index[tt_index] + 2, 2, '<h', file)  # nacita delku listu tt
            self.tt_def_off = find(self.xhdr_index[tt_index] + 4, 4, '<L', file)  # nacita offset definice tt
            self.tt_list_off = find(self.xhdr_index[tt_index] + 8, 4, '<L', file)  # nacita offset listu tt
        else:
            sys.exit("Size and offset of Table tags missing")

        # -----------------------------------------------------------------------
        # Nacteni jmen kanalu
        chan_name_index = self.xhdr_ID.index("0x4e43")
        i = []
        self.chan_name = []
        for d in range(self.xhdr_index[chan_name_index], self.xhdr_index[chan_name_index] + self.nchan * 4):
            a = find(d, 1, '<b', file)
            if a == 0:
                a = 48  # zmena 0 na ascii nulu
            i.append(chr(a))
        # zada nazev kanalu do listu
        for x in range(0, (len(i)), 4):
            h = (i[x] + i[x + 1] + i[x + 2] + i[x + 3])
            h = h.lower()  # prevede nazev na mala pismena
            self.chan_name.append(h)


        #-----------------------------------------------------------------------
        # info o classach tagu
        self.tt_def_nick = []
        self.tt_def_count = []
        tt_def_txtlen = []
        tt_def_txtoff = []
        b = 0
        q = 0
        w = 0
        while b < 32768:
            file.seek(self.tt_def_off + q, 0)
            data = file.read(8)
            (a1, a2, b, c, d) = struct.unpack('<bbHHH', data)
            if a1 == 0: a1 = 48
            if a2 == 0: a2 = 48
            self.tt_def_nick.append(chr(a1) + chr(a2))
            self.tt_def_count.append(b)
            tt_def_txtlen.append(c)
            tt_def_txtoff.append(d)
            q = q + 8
            w += 1
        self.tt_def_count[w - 1] -= 32768

        # nacte indexu jednotlivých tagu
        self.tt_tags = []
        self.tt_tags_class = []
        for d in range(0, self.tt_list_len, 4):
            file.seek(self.tt_list_off + d, 0)
            data = file.read(4)
            (x, y, z, o) = struct.unpack('<BBBB', data)

            self.tt_tags.append(x + 256 * y + 256 * 256 * z)
            self.tt_tags_class.append(o)

        # nacteni textu k classe tagu
        self.tt_def_text = []
        temp = []
        for p in range(len(self.tt_def_nick)):
            for q in range(tt_def_txtlen[p]):
                a = find(self.tt_def_off + tt_def_txtoff[p] + q, 1, '<B', file)
                temp.append(chr(a))
            self.tt_def_text.append("".join(temp))
            temp = []

        # -----------------------------------------------------------------------
        file.close()

    def data_load(self):
        file = open(self.main_file, "rb")
        # nacte vsechny data a kanaly do matice
        file.seek(self.data_org, 0)
        self.data_allchan = np.array(np.fromfile(file, dtype='h', count=self.nsamp * self.nchan))
        self.data_allchan = np.array(np.reshape(self.data_allchan, (self.nchan, self.nsamp), order='F'))
        file.close()

    def tag_info(self):
        # Zobrazeni def tagu vsechno
        for d in range(len(self.tt_def_nick)):
           print(d, "- ", self.tt_def_nick[d], " Count:", self.tt_def_count[d], "         - ", self.tt_def_text[d])

    def chan_info(self):
        # Vypise kanaly s cislama
        for ind, val in enumerate(self.chan_name):
           print(ind, "\t", val)

    def chan_data(self, chan):
        # Vrati data vybraneho kanalu
        chan_number = self.chan_name.index(chan)
        data_1chan = np.array(self.data_allchan[chan_number, :])
        data_chan = np.array((data_1chan - self.zero) * (1 / self.unit))

        return data_chan

    def plot_data(self, data, tag_list):
        # Vykresleni hezkeho grafu, data = list ktery obsahuje data z 1 kanalu,
        # tag_list = list, ktery obsahuje jednotliva jemna tagu ktere chci vykreslit
        # Pozor asi budes muset dat pravou vzorkovacku ne puodni !!!ERROR
        tag = []
        for x in tag_list:
            tag.append(self.tt_def_nick.index(x))

        self.ttd = []
        for k in range(len(self.tt_tags_class)):
            if self.tt_tags_class[k] in tag:
                self.ttd.append(self.tt_tags[k] / self.fsamp)


        zeros = [0] * len(self.ttd)
        t = np.arange(0, (len(data)) / self.fsamp, (len(data) / self.fsamp) / len(data))
        fig, ax = plt.subplots(figsize=(11, 6))
        line = ax.plot(t, data, linewidth=1)
        line4 = ax.plot(self.ttd, zeros, '|', ms=1000)
        line6 = ax.plot(zeros, zeros, linewidth=1)


        plt.setp(line, linewidth=1, color='b', label='Průtok vzduchu')
        plt.setp(line4, color='r')
        plt.setp(line6, color='r', label='Detekované apnoe')


        plt.show()

    def istag(self, tag_list, data_len, segment_len):
        # Udela vektor jehoz velikost je stejna jako velikost počtu segmentu (radky z matice)
        # vsude da 0, –> pokud je segment mezi tagem start a stop tak da 1 –> to vrati
        # data_len = delka dat (po uprave k segmentaci)
        # tag_list = list obsahujci tulples dvojic = [("A+","A-"),("H+","H-")]
        # segment_len = delka segmentu ve vzorkach

        tag_s = []
        tag_e = []

        for tag_twin in tag_list:
            if tag_twin[0] in self.tt_def_nick and tag_twin[1] in self.tt_def_nick: # Kontrola jestli je tag v tabulce pritomen
                tag_s_index = self.tt_def_nick.index(tag_twin[0])
                tag_e_index = self.tt_def_nick.index(tag_twin[1])

                for counter, value in enumerate(self.tt_tags_class):
                    if value == tag_s_index:
                        tag_s.append(self.tt_tags[counter])
                for counter, value in enumerate(self.tt_tags_class):
                    if value == tag_e_index:
                        tag_e.append(self.tt_tags[counter])


        if len(tag_s) != len(tag_e):
            print("Start, end tag count is not the same!")   # Kontrola, jestli jsou start a end tagu stejne mnozstvi
            return []

        # podle tagu oznaceni segmentu
        istag_vect = np.zeros(((int(data_len / segment_len)), 1), dtype=int)
        for x in range(len(tag_s)):
            try:
                istag_vect[int(tag_s[x] / segment_len):int(tag_e[x] / segment_len)] = 1
            except:
                pass


        # if sum(istag_vect) == 0:
        #     return []

        return istag_vect

    def istag_slide(self, tag_list, data_seg, times):

        tag_s = []
        tag_e = []


        for tag_twin in tag_list:
            if tag_twin[0] in self.tt_def_nick and tag_twin[1] in self.tt_def_nick: # Kontrola jestli je tag v tabulce pritomen
                tag_s_index = self.tt_def_nick.index(tag_twin[0])
                tag_e_index = self.tt_def_nick.index(tag_twin[1])

                for counter, value in enumerate(self.tt_tags_class):
                    if value == tag_s_index:
                        tag_s.append(self.tt_tags[counter])
                for counter, value in enumerate(self.tt_tags_class):
                    if value == tag_e_index:
                        tag_e.append(self.tt_tags[counter])


        if len(tag_s) != len(tag_e):
            print("Start, end tag count is not the same!")   # Kontrola, jestli jsou start a end tagu stejne mnozstvi
            return []

        num_of_segments = int(np.shape(data_seg)[0])
        num_of_tags     = int(np.shape(tag_s)[0])

        t.tic()
        # podle tagu oznaceni segmentu
        istag_vect = np.zeros(((num_of_segments), 1), dtype=int)
        tag_in_count = np.zeros(((num_of_segments), 1), dtype=int)
        seg_len_times = 0.8*(times[10,1] - times[10,0])

        for x in range(num_of_segments):
            for y in range(num_of_tags):
                if    tag_s[y] > times[x,0] and tag_e[y] < times[x,1] :
                    tag_in_count[x, 0] +=1
                    if (tag_e[y] - tag_s[y]) > seg_len_times:
                        istag_vect[x, 0] += 1

                elif  tag_s[y] < times[x,0] and tag_e[y] < times[x,1] and tag_e[y] > times[x,0] :
                    tag_in_count[x, 0] += 1
                    if (tag_e[y] - times[x,0]) > seg_len_times:
                        istag_vect[x, 0] += 1

                elif  tag_s[y] > times[x,0] and tag_e[y] > times[x,1] and tag_s[y] < times[x,1] :
                    tag_in_count[x, 0] += 1
                    if (times[x,1] - tag_s[y]) > seg_len_times:
                        istag_vect[x, 0] += 1

                elif  tag_s[y] < times[x,0] and tag_e[y] > times[x,1] :
                    tag_in_count[x, 0] += 1
                    if (times[x,1] - times[x,0]) > seg_len_times:
                        istag_vect[x, 0] += 1
                else:
                    pass
            if istag_vect[x, 0] > 1 or tag_in_count[x,0] > 1:
                istag_vect[x, 0] == 0

        # Kontrola pro zdvojene tagy
        for x in range(np.shape(istag_vect)[0]):
            if istag_vect[x, 0] != 0 and istag_vect[x, 0] != 1:
                istag_vect[x, 0] = 1

        t.toc()

        self.tag_orig_s = tag_s
        self.tag_orig_e = tag_e

        return istag_vect

    def anonymous(self):
        # Anonymizuje celý d_file (upraví patient ID na 3333

        file = open(self.file_name, "r+b")

        xtype = "0x4944"                # V tomto extendet hedru je jmeno pacienta
        if xtype in self.xhdr_ID:

            pid_index = self.xhdr_ID.index(xtype)
            for q in range(self.xhdr_size[pid_index]):
                save(self.xhdr_index[pid_index] + q, '<b', 3, file)

        file.close()
        print("...Anonymous were here...\n")

    def preprocess_flow(self):

        self.data_load()  # Nacte data
        flow_orig = self.chan_data("flow")  # Nacteni jednoho kanalu (flow)
        flow = flow_orig

        # Navrhne filtr, vypocita koeficienty a vyfiltruje signal               - nejspis neni potreba, fultruje se jeste jednou pri podvzorkovani
        sos = sig.butter(10, 5 / (self.fsamp / 2), btype='lowpass', output='sos')
        flow = scipy.signal.sosfilt(sos, flow)

        # Vykresli ukazku nefiltrovaneho a filtrovaneho signalu
        # f, axarr = plt.subplots(2, sharex=False)
        # axarr[0].plot(flow_orig)
        # axarr[1].plot(flow)
        # plt.show()

        # Segmentace
        seg_len_sec = 10  # Delka segmentu v sec
        seg_len = self.fsamp * seg_len_sec  # Delka segmentu ve vzorkach

        if (flow.size % seg_len) != 0:
            flow = flow[:-(flow.size % seg_len)].copy()  # Uprava delky kuli segmentaci (zkraceni))
        flow_seg = np.reshape(flow, (int(flow.size / seg_len), seg_len))  # Segmentace fixni

        # Prideni matice zacatek/konec segmentu ke kazdemu segnemtu
        times = np.zeros((np.shape(flow_seg)[0],2), dtype=int)
        for x in range(np.shape(flow_seg)[0]):
            if x == 0:
                times[x, 0] = 0
                times[x, 1] = times[x, 0]   + seg_len
            else:
                times[x,0] = times[x-1,0]   + seg_len + 1
                times[x,1] = times[x,0]     + seg_len

        # Urceni pritomnisti apnoe v segmentu
        tags = self.istag([("O+", "O-"), ("A+", "A-")], len(flow), seg_len)  # Udela vektor s oznacenim segmentu
        if tags == []:
            raise Exception("Tags problem!")

        # Spojeni tagu o apnoe s časovou značkou
        tags = np.hstack((tags, times))

        # Podvzorkovani 5x
        flow_seg_down = sig.decimate(flow_seg, 5, zero_phase=True)
        seg_len = (np.shape(flow_seg_down)[1])  # Delka segmentu ve vzorkach

        # print("Shape tags\t\t: "+str(np.shape(tags)))
        # print("Shape flow_seg\t: "+str(np.shape(flow_seg_down)))             # Velikost (rady=segmenty, sloupce)

        #print("Pred Tuckey: " + str(np.shape(flow_seg_down)))

        # Vypocet outline pomoci IQR Tuckyho metody
        flow_down = flow_seg_down.flatten()  # Udela z matice vektor
        quartile_1, quartile_3 = np.percentile(flow_down, [25, 75])  # Vypocita percentil
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)

        #print("Upper: " + str(upper_bound))
        #print("Lower: " + str(lower_bound))


        # Odstraneni segmentu ktere obsahujou outlier
        data_flow_seg = []
        data_tags = []
        for x in range(np.shape(flow_seg_down)[0]):
            if max(flow_seg_down[x, :]) < upper_bound:
                if min(flow_seg_down[x, :]) > lower_bound:
                    data_flow_seg.append(flow_seg_down[x, :])
                    data_tags.append(tags[x])

        data_flow_seg = np.asarray(data_flow_seg)  # konverze na numpy array
        data_tags = np.asarray(data_tags)  # konverze na numpy array

        #print("Po Tuckey: " + str(np.shape(data_flow_seg)))

        # Normalizece pomoci Z-skore
        data_flow_seg = stats.zscore(data_flow_seg, axis=None)


        return data_flow_seg, data_tags

    def preprocess_spo2(self):
        # Nacte kanal spo2 a vyhodi ho nasegmentovanej

        self.data_load()  # Nacte data
        spo2_orig = self.chan_data("spo2")  # Nacteni jednoho kanalu (spo2)

        sos = sig.butter(2, 0.06 / (self.fsamp / 2), btype='lowpass', output='sos')
        spo2_filt = scipy.signal.sosfilt(sos, spo2_orig)


        # Nasobeni
        spo2 = np.multiply(np.array(spo2_filt), 10)

        # Preprocess - segmentace a vzorkovani
        seg_len_sec = 10  # Delka segmentu v sec
        seg_len = self.fsamp * seg_len_sec  # Delka segmentu ve vzorkach

        if (spo2.size % seg_len) != 0:
            spo2 = spo2[:-(spo2.size % seg_len)].copy()  # Uprava delky kuli segmentaci (zkraceni))
        spo2_seg = np.reshape(spo2, (int(spo2.size / seg_len), seg_len))  # Segmentace fixni

        # Prideni matice zacatek/konec segmentu ke kazdemu segnemtu
        times = np.zeros((np.shape(spo2_seg)[0], 2), dtype=int)
        for x in range(np.shape(spo2_seg)[0]):
            if x == 0:
                times[x, 0] = 0
                times[x, 1] = times[x, 0] + seg_len
            else:
                times[x, 0] = times[x - 1, 0] + seg_len + 1
                times[x, 1] = times[x, 0] + seg_len

        # Urceni pritomnisti apnoe v segmentu
        tags = self.istag([("S+", "S-")], len(spo2), seg_len)  # Udela vektor s oznacenim segmentu
        if tags == []:
            raise Exception("Tags problem!")

        # Spojeni tagu o apnoe s časovou značkou
        tags = np.hstack((tags, times))

        spo2_out = []
        tags_out = []
        # hlidani vypadku cidla
        for x in range(tags.shape[0]):
            if max(spo2_seg[x,:]) < 101 and min(spo2_seg[x,:]) > 70:
                spo2_out.append(spo2_seg[x,:])
                tags_out.append(tags[x,:])

        spo2_out = np.asarray(spo2_out)  # konverze na numpy array
        tags = np.asarray(tags_out)  # konverze na numpy array

        # Podvzorkovani 5x
        spo2_seg_down = sig.decimate(spo2_out, 5, zero_phase=True)
        self.seg_len = (np.shape(spo2_seg_down)[1])  # Delka segmentu ve vzorkach

        spo2_seg_down = stats.zscore(spo2_seg_down, axis=None)
        print(spo2_seg_down.shape)

        return spo2_seg_down, tags

    def preprocess_flow_slide(self, chan_name):

        # Nacte kanal spo2 a vyhodi ho nasegmentovanej
        prekryv = 9                         # kolik sec signalu se bude prekryvat
        seg_len_sec = 10                    # Delka segmentu v sec

        # Nacteni dat
        self.data_load()  # Nacte data
        spo2_orig = self.chan_data(chan_name)  # Nacteni jednoho kanalu (spo2)
        #spo2_orig = spo2_orig[0:500000]

        seg_len = self.fsamp * seg_len_sec   # Delka segmentu ve vzorkach
        bucket_size = seg_len                # Velikost segmentu
        overlap_count = prekryv * self.fsamp # Prekryv

        # Vytvoreni casovych znacek segmentu
        time_s = list(range(0, len(spo2_orig)+1, bucket_size - overlap_count))
        time_e = list(range(bucket_size - 1, len(spo2_orig)+bucket_size, bucket_size - overlap_count))

        # Podvzorkovani
        spo2_down = sig.decimate(spo2_orig, 5, zero_phase=True)
        fsamp_new = int(self.fsamp/5)

        # Prenastaveni hodnot po podvzorkovani
        seg_len = fsamp_new * seg_len_sec    # Delka segmentu ve vzorkach
        bucket_size = seg_len                # Velikost segmentu
        overlap_count = prekryv * fsamp_new  # Prekryv

        # Filtrace
        sos = sig.butter(10, 5 / (fsamp_new / 2), btype='lowpass', output='sos')
        spo2 = scipy.signal.sosfilt(sos, spo2_down)

        # Segmentace sliging window
        slider = Slider(bucket_size, overlap_count)
        slider.fit(spo2)
        print("Segmentation flow")
        #data = np.empty((bucket_size))
        data = []

        while True:
            window_data = slider.slide()
            if len(window_data) < bucket_size:
                break
            data.append(window_data)
            if slider.reached_end_of_list(): break

        # Konverze na numpy array
        spo2_seg = np.array(data)

        # Prideni matice zacatek/konec segmentu ke kazdemu segnemtu
        times = np.zeros((np.shape(spo2_seg)[0], 2), dtype=int)
        for x in range(np.shape(spo2_seg)[0]):
            times[x, 0] = time_s[x]
            times[x, 1] = time_e[x]

        # Urceni pritomnisti apnoe v segmentu
        tags = self.istag_slide([("O+", "O-"), ("A+", "A-")], spo2_seg, times)  # Udela vektor s oznacenim segmentu
        if tags == []:
            raise Exception("Tags problem!")

        # Spojeni tagu o apnoe s časovou značkou
        tags = np.hstack((tags, times))


        flow_down = spo2_seg.flatten()  # Udela z matice vektor
        quartile_1, quartile_3 = np.percentile(flow_down, [25, 75])  # Vypocita percentil
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)

        # print("Upper: " + str(upper_bound))
        # print("Lower: " + str(lower_bound))


        # Odstraneni segmentu ktere obsahujou outlier
        # data_flow_seg = []
        # data_tags = []
        # for x in range(np.shape(spo2_seg)[0]):
        #     if max(spo2_seg[x, :]) < upper_bound:
        #         if min(spo2_seg[x, :]) > lower_bound:
        #             data_flow_seg.append(spo2_seg[x, :])
        #             data_tags.append(tags[x,:])
        #
        # data_flow_seg = np.asarray(data_flow_seg)  # konverze na numpy array
        # data_tags = np.asarray(data_tags)  # konverze na numpy array

        # Nahrada bez odstraneni outliers
        data_tags = tags
        data_flow_seg = spo2_seg


        spo2_seg_down = stats.zscore(data_flow_seg, axis=None)


        return data_flow_seg, data_tags

    def preprocess_spo2_slide(self):
        # Nacte kanal spo2 a vyhodi ho nasegmentovanej
        prekryv = 9                         # kolik sec signalu se bude prekryvat
        seg_len_sec = 10                    # Delka segmentu v sec

        # Nacteni dat
        self.data_load()  # Nacte data
        spo2_orig = self.chan_data("spo2")  # Nacteni jednoho kanalu (spo2)
        #spo2_orig = spo2_orig[0:500000]

        # Nastaveni

        seg_len = self.fsamp * seg_len_sec  # Delka segmentu ve vzorkach

        bucket_size = seg_len               # Velikost segmentu
        overlap_count = prekryv * self.fsamp      # Prekryv

        # Vytvoreni casovych znacek segmentu
        time_s = list(range(0, len(spo2_orig)+1, bucket_size - overlap_count))
        time_e = list(range(bucket_size - 1, len(spo2_orig)+bucket_size, bucket_size - overlap_count))

        # Podvzorkovani
        spo2_down = sig.decimate(spo2_orig, 5, zero_phase=True)
        fsamp_new = int(self.fsamp/5)

        # Prenastaveni hodnot po podvzorkovani
        seg_len = fsamp_new * seg_len_sec  # Delka segmentu ve vzorkach
        bucket_size = seg_len               # Velikost segmentu
        overlap_count = prekryv * fsamp_new      # Prekryv

        # Filtrace
        sos = sig.butter(2, 0.06 / (fsamp_new / 2), btype='lowpass', output='sos')
        spo2_filt = scipy.signal.sosfilt(sos, spo2_down)

        # Nasobeni
        spo2 = np.multiply(np.array(spo2_filt), 10)

        # Segmentace sliging window
        slider = Slider(bucket_size, overlap_count)
        slider.fit(spo2)
        print("Segmentation SpO2")
        #data = np.empty((bucket_size))
        data = []

        while True:
            window_data = slider.slide()
            if len(window_data) < bucket_size:
                break
            #data = np.append(data, window_data, axis=0)
            data.append(window_data)
            if slider.reached_end_of_list(): break

            # if np.shape(data)[0]%100 == 0:
            #     print(str(int(np.shape(data)[0]/bucket_size)) + "/" + str(int(len(spo2)/(bucket_size-overlap_count))))
            #     print(str(int(sys.getsizeof(data)/1000000))+" MB")

        # print(np.shape(data))
        # print(type(data))
        #
        # print("Reshaping")
        # spo2_seg = np.reshape(data[0:(len(data) - (len(data) % bucket_size))],
        #                   (int(len(data) / bucket_size), bucket_size), order='C')

        # Konverze na numpy array
        spo2_seg = np.array(data)


        # Prideni matice zacatek/konec segmentu ke kazdemu segnemtu
        times = np.zeros((np.shape(spo2_seg)[0], 2), dtype=int)
        for x in range(np.shape(spo2_seg)[0]):
            times[x, 0] = time_s[x]
            times[x, 1] = time_e[x]

        # Urceni pritomnisti apnoe v segmentu
        tags = self.istag_slide([("S+", "S-"), ("D+", "D-")], spo2_seg, times)  # Udela vektor s oznacenim segmentu
        if tags == []:
            raise Exception("Tags problem!")

        # Spojeni tagu o apnoe s časovou značkou
        tags = np.hstack((tags, times))


        # # hlidani vypadku cidla
        # spo2_out = []
        # tags_out = []
        # for x in range(tags.shape[0]):
        #     if max(spo2_seg[x, :]) < 101 and min(spo2_seg[x, :]) > 70:
        #         spo2_out.append(spo2_seg[x, :])
        #         tags_out.append(tags[x, :])
        #
        # spo2_out = np.asarray(spo2_out)  # konverze na numpy array
        # tags = np.asarray(tags_out)  # konverze na numpy array

        spo2_out = spo2_seg

        spo2_seg_down = stats.zscore(spo2_out, axis=None)

        return spo2_seg_down, tags

    def save_tags(self, tags, s_class, e_class, s_tag, e_tag):
        # Ulozeni tagu do .d filu
        # Prepsani spatnych informaci----------------------------------------------------------------
        file = open(self.file_name, "r+b")


        tag_offset_count = np.shape(tags)[0] * 4 + self.tt_list_len

        # upraveni velikosti a offsetu Table tags
        tt_index = self.xhdr_ID.index("0x5454")
        save(self.xhdr_index[tt_index] + 2, '<H', tag_offset_count, file)  # nacita delku listu tt

        # Pocitani celkoveho mnozstvi jednotlivich tagu
        xs = 0
        xe = 0
        for tag in tags[:,0]:
            if tag == s_tag:
                xs += 1
            elif tag == e_tag:
                xe += 1

        # upraveni poctu tagu
        q = s_class  # cislo kanalu
        save(self.tt_def_off + 0 + q * 8, '<B', ord(s_tag[0]), file)
        save(self.tt_def_off + 1 + q * 8, '<B', ord(s_tag[1]), file)
        save(self.tt_def_off + 2 + q * 8, '<H', xs,       file)    # Bacha pocet delam delenim vsech
        q = e_class  # cislo kanalu
        save(self.tt_def_off + 0 + q * 8, '<b', ord(e_tag[0]), file)
        save(self.tt_def_off + 1 + q * 8, '<b', ord(e_tag[1]), file)
        save(self.tt_def_off + 2 + q * 8, '<H', xe,       file)


        file.close()

        # Pridani tagu na konec----------------------------------------------------------------------
        file = open(self.file_name, "r+b")
        file.seek(self.tt_list_off + self.tt_list_len, 0)

        for o in range(np.shape(tags)[0]):
            # Zapsani tagu pro zacatky apnoe

            if   tags[o,0] == s_tag:
                tag_class = s_class  # Urcuje jakou tridu dostane tag
            elif tags[o,0] == e_tag:
                tag_class = e_class  # Urcuje jakou tridu dostane tag

            tag_val = int(tags[o,1])
            tag_val_b = tag_val.to_bytes(3, byteorder='little')
            file.write(tag_val_b)
            tag_class_b = struct.pack('<B', tag_class)
            file.write(tag_class_b)


        file.close()
        return ()

def tag_seg2tags(tag_seg, s_tag, e_tag):
    # Udela z formatu: "tag      | start | stop"
    # Pokud bude vic apnoe segmentu za sebou, tak je spoji
    # Format ->        "znacka   | misto"
    tag_char = []
    tag_place = []
    for x in range(0,np.shape(tag_seg)[0]-1):
        # Prida start tag
        if      tag_seg[x,0] == 0 and tag_seg[x+1,0] == 1:
            tag_char.append(s_tag)
            tag_place.append(tag_seg[x+1,1])
        # Prida end tag
        elif    tag_seg[x,0] == 1 and tag_seg[x+1, 0] == 0:
            tag_char.append(e_tag)
            tag_place.append(tag_seg[x,2])

    tag_char = np.array(tag_char)
    tag_place = np.array(tag_place)

    tags = np.vstack((tag_char, tag_place)).T

    return tags

def tag_seg2tags_slide(tag_seg, s_tag, e_tag):
    # Udela z formatu: "tag      | start | stop" (celej segment, slide)
    # Pokud bude vic apnoe segmentu za sebou, tak je spoji
    # Format ->        "znacka   | misto"


    tag_char = []
    tag_place = []
    stst = 0
    for x in range(5,np.shape(tag_seg)[0]-5):
        # Prida start tag
        if  tag_seg[x,0] == 0 and tag_seg[x+1,0] == 1 and tag_seg[x+2,0] == 1 and tag_seg[x+3,0] == 1 and tag_seg[x+4,0] == 1 and tag_seg[x+5,0] == 1:
            tag_char.append(s_tag)
            tag_place.append(tag_seg[x+1,1])

        # Prida end tag
        if  tag_seg[x-5,0] == 1 and tag_seg[x-4,0] == 1 and tag_seg[x-3,0] == 1 and tag_seg[x-2,0] == 1 and tag_seg[x-1,0] == 1 and tag_seg[x, 0] == 0:
            tag_char.append(e_tag)
            tag_place.append(tag_seg[x-1,2])

    if tag_char.count(s_tag) != tag_char.count(e_tag):
        sys.exit("ERROR, tag_seg2tags_slide, s&e tags not same")

    tag_char = np.array(tag_char)
    tag_place = np.array(tag_place)

    tags = np.vstack((tag_char, tag_place)).T

    return tags

def compare2tag_sets(orig_tag_s, orig_tag_e, new_tag):
    # Bere originalni tagy a nove tagy ve formatu:
    # [char | place ].....tzn. pr.: ["o+" | 556264]
    #                               ["o-" | 542424]
    # teda tak bere new_tag, orig tagy bere jako dva vektory s pozicema s,e

    # Jako spravne to bere pokud se prekryva 80% apnoe a tagu
    char_tag_start = new_tag[0,0]

    new_tag_s = []
    new_tag_e = []
    for x in range(np.shape(new_tag)[0]):
        if new_tag[x,0] == char_tag_start:
            new_tag_s.append(int(new_tag[x,1]))
        else:
            new_tag_e.append(int(new_tag[x,1]))

    TP = 0
    FP = 0
    FN = 0
    TP_bool = False

    seg_len_times = 0.8 * (orig_tag_e[1] - orig_tag_s[1])

    #_ for x in range(len(new_tag_s)):
    #_     for y in range(len(orig_tag_s)):
    for y in range(len(orig_tag_s)):
        for x in range(len(new_tag_s)):

            if orig_tag_s[y] > new_tag_s[x] and orig_tag_e[y] < new_tag_e[x]:
                if (orig_tag_e[y] - orig_tag_s[y]) > seg_len_times:
                    TP += 1
                    TP_bool = True
                    break

            elif orig_tag_s[y] < new_tag_s[x] and orig_tag_e[y] < new_tag_e[x]:
                if (orig_tag_e[y] - new_tag_s[x]) > seg_len_times:
                    TP += 1
                    TP_bool = True
                    break

            elif orig_tag_s[y] > new_tag_s[x] and orig_tag_e[y] > new_tag_e[x]:
                if (new_tag_e[x] - orig_tag_s[y]) > seg_len_times:
                    TP += 1
                    TP_bool = True
                    break

            elif orig_tag_s[y] < new_tag_s[x] and orig_tag_e[y] > new_tag_e[x]:
                if (new_tag_e[x] - new_tag_s[x]) > seg_len_times:
                    TP += 1
                    TP_bool = True
                    break

        if TP_bool == False:
            FN += 1
        TP_bool = False

    FP = np.shape(new_tag_s)[0] - TP

    return TP, FP, FN














