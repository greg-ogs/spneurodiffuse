# import time
import pandas as pd
import numpy as np
import os
import mysql.connector as mysql


# from tqdm.auto import tqdm

class Datasets:
    def __init__(self):
        self.yv = None
        self.xv = None
    @staticmethod
    def create_base_dataset():  # Create a dataset using all possible combinations
        # High RAM memory, low time
        print("Creating base dataset...")
        x = np.arange(0, 25, 0.1)  # Creates an array from 0 to 25 for x
        y = np.arange(0, 25, 0.1)  # Creates an array from 0 to 25 for y
        coords = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        coords = np.round(coords, 3)
        n_dataset = pd.DataFrame(coords)
        n_dataset.head(100)
        n_dataset.to_csv("base_dataset.csv", index=False)
        print("Base dataset created. Please run again")

        # Low memory but long time
        # n_dataset = [[10, 10]]
        # x = np.arange(0, 25, 0.001)
        # y = np.arange(0, 25, 0.001)
        # coords_generator = ((xi, yi) for xi in x for yi in y)  # Generator object is returned
        # for coordinate in tqdm(coords_generator, desc="Creating base dataset", ascii=True, total=625000000):
        #     n_dataset = np.vstack((n_dataset, np.asarray(coordinate)))
        # print("Base dataset created.")

    @staticmethod
    def create_dynamic_range(coord):

        if coord >= 1: lxd = coord - 1; dr = 0
        else: lxd = 0; dr = 1-coord;
        if coord <= 24: lxu = coord + 1; ur = 0
        else: lxu = 25; ur = (coord + 1) - 25;
        lxd = lxd - ur
        lxu = lxu + dr

        xv = np.arange(lxd, lxu, 0.01)
        return xv

    def concatenate_coordinates(self, x, y):
        self.xv = self.create_dynamic_range(x)
        self.yv = self.create_dynamic_range(y)

        print('X vector: {}'.format(self.xv))
        print('Y vector: {}'.format(self.yv))

        aux = 0
        for coordinates in zip(self.xv, self.yv):
            print("Pair of coordinates: {}, {}".format(round(coordinates[0], 3), round(coordinates[1], 3)))
            aux = aux + 1
            # add mysql query to insert the columns

        print(aux)
        if x <= 1 or y <= 1 or x >= 24 or y >= 24:
            print("Pair of real coordinates: {}, {}".format(round(x, 3), round(y, 3)))
            # add mysql query to insert the original pair of coordinates several times
            pass

class ReadDataset:
    def __init__(self):
        self.dataset = None
        self.file_path = None

        # MySQL
        self.mydb = mysql.connect(
            host="172.17.0.2",
            user="user",
            database="dataset",
            password="userpass", port=3306
        )

        self.mycursor = self.mydb.cursor()

    # initial adjusment
    def load_dataset_from_csv(self):
        try:
            self.file_path = 'base_dataset.csv'
            self.dataset = pd.read_csv(self.file_path)
            print(self.dataset.shape)

        except FileNotFoundError:

            print(f"Error: File '{self.file_path}' not found.")

        except pd.errors.EmptyDataError:

            print(f"Error: File '{self.file_path}' is empty.")

        except pd.errors.ParserError:

            print(f"Error: Parsing error while reading '{self.file_path}'.")

        except Exception as e:  # Catch-all for other unexpected errors

            print(f"Unexpected error while reading '{self.file_path}': {e}")

    def insert_bool(self, x_d, y_d, x_u, y_u):
        query = "UPDATE base_dataset_low_res SET RESULT_3 = %s WHERE C1 > %s AND C2 > %s AND C1 < %s AND C2 < %s ;"
        val = (1, x_d, y_d, x_u, y_u)
        self.mycursor.execute(query, val)
        self.mydb.commit()


if __name__ == '__main__':
    newDataset = Datasets()
    newDataset.concatenate_coordinates(12.5, 24.6)

    # dataset = ReadDataset()
    # To update data into mysql server table
    # x = 17.9
    # y = 12.8
    # xd = round(x - 0.3, 1)
    # yd = round(y - 0.3, 1)
    # xu = round(x + 0.3, 1)
    # yu = round(y + 0.3, 1)
    # dataset.insert_bool(xd, yd, xu, yu)
