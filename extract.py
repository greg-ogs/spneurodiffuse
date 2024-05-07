import mysql.connector
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class SqlQueryE:
    def __init__(self):
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="greg",
            password="contpass01",
            database="airy"
        )

        self.mycursor = self.mydb.cursor()

    def save_query(self):
        self.mycursor.execute("SELECT * FROM MAPING")
        myresult = self.mycursor.fetchall()
        aux = np.asarray(myresult[0])
        for i in range(1, len(myresult)):
            array_0 = np.asarray(myresult[i])
            aux = np.vstack((aux, array_0))
        df = pd.DataFrame(aux, columns=['X', 'Y', 'ID'])
        df.to_csv('map.csv', index=False)

        self.mycursor.execute("SELECT * FROM times")
        myresult_0 = self.mycursor.fetchall()
        aux = np.asarray(myresult_0[0])
        for i in range(1, len(myresult_0)):
            array_0 = np.asarray(myresult_0[i])
            aux = np.vstack((aux, array_0))
        df = pd.DataFrame(aux, columns=['ID', 'times'])
        df.to_csv('times.csv', index=False)


class PlottingData:
    def __init__(self):
        self.table_times = pd.read_csv('times.csv')
        # self.table = pd.read_csv('C:/Users/grego/Downloads/GitHub/hw.CSV')
        # print(self.table)
        sns.set_theme(style="darkgrid")

    def plot_table(self, column_y, title):
        sns.relplot(
            data=self.table,
            x="Step", y=column_y,
            palette="Paired",
            hue="Iteration",
            kind="line",
            units="Iteration", estimator=None
            # markers=True,
            # dashes=False
        )
        plt.title(title)
        name = title + '_plot.png'
        plt.savefig(name, dpi=1000)
        plt.show()

    def plot_times(self):
        sns.relplot(
            data=self.table_times,
            x="ID", y='times',
            palette="viridis",
            hue="control",
            # kind="line",
            # units="Iteration", estimator=None
            # markers=True,
            # dashes=False
        )
        title = 'times'
        plt.title(title)
        name = title + '_plot.png'
        plt.savefig(name, dpi=1000)
        plt.show()

# from skimage.data import astronaut
# from skimage.color import rgb2gray
# from skimage.filters import sobel
# from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
# from skimage.segmentation import mark_boundaries
# from skimage.util import img_as_float
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# class super_pixels:
#     def __init__(self, image):
#         self.img = img_as_float(Image.open(image))
#
#     def s_pixel(self):
#         self.segments = slic(self.img, n_segments=800, compactness=10, sigma=1, start_label=1)
#
#     def center_of_the_beam(self):
#         nlabels = np.amax(self.segments)
#         nlabels = nlabels + 1
#         # print(nlabels)
#         nlabels = int(nlabels)
#         values = []
#         for i in range(1, nlabels):
#             coor = np.where(self.segments == i)  #coordenada de cada segmento
#             #co = [coor[0][0],coor[1][0]]# toma la primera coordenada de cada segmento
#             #segmentVal = image[co[0]][co[1]][2]#usa la coordenada anterior para buscar el valor en la imagen
#             arraysize = coor[0].shape  # canntidad de coordenadas de el segmento actual
#             # print(str(arraysize) + 'cantidad de coordenadas en el setgmeto ' + str(i))
#             arrsiz = arraysize[0]
#             meansum = []
#             for j in range(arrsiz):
#                 #individualCoor = [coor[0][j],coor[1][j]]#coordenada individual de cda pixel del segemento
#                 coorVal = self.img[coor[0][j]][coor[1][j]][0]  #valaor de cada pixel del segmento
#                 meansum.append(coorVal)  #se agrega el valor a un vector
#             segmentVal = np.mean(meansum)  #media
#             values.append(segmentVal)  #agrega ese valor a una variable (la media de cada segmento)
#             # print('valor del segmento ' + str(segmentVal))
#         # print('vector de valores' + str(values))
#         maxsegment = np.where(values == np.amax(values))  #elige segmento con valor maximo
#         maxS = maxsegment[0] + 1  # compensacion del 0 en el indice del array
#         maxseg = maxS[0]
#         # print(str(maxseg) + ' es el segmento con valor maximo')
#         maxVC = np.where(self.segments == maxseg)  #selecciona todas las coordenadas del segmento con valor maximo
#         # print(maxVC)
#         #calcular la distancia desde el segmento hasta el centro
#         arraysz = maxVC[0].shape  #dimencion del conjunto de coordenadas del segmento
#         arsz = int(arraysz[0] / 2)  #la mitad de ese conjunto
#         XselectCoor = maxVC[1][arsz]  #coordenada intermedia en x
#         X = XselectCoor
#         YselectCoor = maxVC[0][arsz]  #coordenada intermedia en y
#         Y = YselectCoor
#         return X, Y


# query_e = SqlQueryE()
# query_e.save_query()
plot = PlottingData()
# plot.plot_table("Physical Memory Load [%]", "Physical Memory Load over Time")
# plot.plot_table("CPU Package Power [W]", "CPU Package Power over Time")
# plot.plot_table("Total CPU Utility [%]", "Total CPU Utility over Time")
plot.plot_times()

# vex = [0]*39
# vey = [0]*39
# for i in range(39):
#     image = "/content/drive/MyDrive/error/ERROR_" + str(i) + ".png"
#     # print(image)
#     seg_c = super_pixels(image)
#     seg_c.s_pixel()
#     X, Y = seg_c.center_of_the_beam()
#     eX = X-187
#     vex[i] = eX
#     eY = Y-150
#     vey[i] = eY
#     print("Error x = " + str(eX))
#     print("Error y = " + str(eY))
# print(np.mean(vex))
# print(np.mean(vey))