import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn.preprocessing import StandardScaler


class super_pixels:
    def s_pixel(self):
        # deprecated
        numSegments = 300  # Segments for superpixels
        # apply SLIC and extract the segments
        self.segments = slic(self.image, n_segments=numSegments, sigma=5)
        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(self.image, self.segments))
        plt.axis("off")
        plt.show()

    def center_of_the_beam(self):
        # Center of the beam code
        nlabels = np.amax(self.segments)  # Real segments num
        nlabels = nlabels + 1  # Index 0 compensation
        nlabels = int(nlabels)
        values = []
        for i in range(1, nlabels):
            coor = np.where(self.segments == i)  # Select the pixels that correspond to segment "i"
            # co = [coor[0][0],coor[1][0]]# toma la primera coordenada de cada segmento
            # segmentVal = image[co[0]][co[1]][2]#usa la coordenada anterior para buscar el valor en la imagen
            arraysize = coor[0].shape  # Num of pixels in this particular segment
            arrsiz = arraysize[0]  # tuple to int
            meansum = []
            for j in range(arrsiz):
                coorVal = self.image[coor[0][j]][coor[1][j]][0]  # Value per pixel between 0-255
                meansum.append(coorVal)  # Mean value per segment, vector
            segmentVal = np.mean(meansum)  # Mean value per segment (segment "i")
            values.append(segmentVal)  # Append the mean value for segment number "i" for future comparation
        maxsegment = np.where(values == np.amax(values))  # Max mean value in the nlabels segments, return index array
        maxS = maxsegment[0] + 1  # Index 0 compensation
        maxseg = maxS[0]
        maxVC = np.where(
            self.segments == maxseg)  # selecciona todas las coordenadas del segmento con valor maximo
        # calcular la distancia desde el segmento hasta el centro
        arraysz = maxVC[0].shape  # dimencion del conjunto de coordenadas del segmento
        arsz = int(arraysz[0] / 2)  # Punto medio del conjunto de coordenadas del segmento
        self.XselectCoor = maxVC[1][arsz]  # coordenada intermedia del segmento con mas intencidad en x
        self.YselectCoor = maxVC[0][arsz]  # coordenada intermedia del segmento con mas intencidad en y

super_img = super_pixels()
super_img.s_pixel()