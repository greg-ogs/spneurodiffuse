# import time
import pandas as pd
import numpy as np
import os
import mysql.connector as mysql


# from tqdm.auto import tqdm


def create_base_dataset():
    # High RAM memory, low time
    print("Creating base dataset...")
    x = np.arange(0, 25, 0.01)  # Creates an array from 0 to 25 for x
    y = np.arange(0, 25, 0.01)  # Creates an array from 0 to 25 for y
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


class Dataset:
    def __init__(self):
        self.dataset = None
        self.file_path = None

        # MySQL
        self.mydb = mysql.connect(
            host="172.17.0.3",
            user="user",
            database="dataset",
            password="userpass", port=3306
        )

        self.mycursor = self.mydb.cursor()

    # initial adjusment
    def load_dataset(self):
        try:
            self.file_path = 'base_dataset.csv'
            self.dataset = pd.read_csv(self.file_path)
            print(self.dataset.shape)

        except FileNotFoundError:

            print(f"Error: File '{self.file_path}' not found.")
            create_base_dataset()

        except pd.errors.EmptyDataError:

            print(f"Error: File '{self.file_path}' is empty.")

        except pd.errors.ParserError:

            print(f"Error: Parsing error while reading '{self.file_path}'.")

        except Exception as e:  # Catch-all for other unexpected errors

            print(f"Unexpected error while reading '{self.file_path}': {e}")

    def insert_bool(self, x_d, y_d, x_u, y_u, el_t):
        query = ("UPDATE base_dataset SET RESULT_2 = %s, TIME_ELAPSED_2_seconds = %s WHERE C1 > %s AND C2 > %s AND"
                 " C1 < %s AND C2 < %s ;")

        val = (1, el_t, x_d, y_d, x_u, y_u)
        self.mycursor.execute(query, val)
        self.mydb.commit()


if __name__ == '__main__':
    dataset = Dataset()
    # dataset.load_dataset()
    x = 16.81
    y = 12.01
    elt = int(np.round((np.random.rand() * (864000 - 86400 + 1) + 86400), 0))
    xd = round(x - 0.1, 2)
    yd = round(y - 0.1, 2)
    xu = round(x + 0.1, 2)
    yu = round(y + 0.1, 2)
    dataset.insert_bool(xd, yd, xu, yu, elt)
