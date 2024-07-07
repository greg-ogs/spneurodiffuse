# import time
import pandas as pd
import numpy as np
import os


# from tqdm.auto import tqdm


def create_base_dataset():
    # High RAM memory, low time
    print("Creating base dataset...")
    x = np.arange(0, 25, 0.001)  # Creates an array from 0 to 25 for x
    y = np.arange(0, 25, 0.001)  # Creates an array from 0 to 25 for y
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


class Dataset():
    def __init__(self):
        pass

    # initial adjusment
    def load_dataset(self):
        try:
            self.file_path = 'base_dataset.csv'
            # chunksize = 100000
            # it = 0
            # start_time = time.time()
            # for chunk in pd.read_csv(self.file_path, chunksize=chunksize):
            #     it = it + 1
            #     current_time = time.time()  # Time after each chunk
            #     elapsed_time = current_time - start_time
            #     print(f"Elapsed Time: {elapsed_time:.2f} seconds | Rows Processed: {chunksize*it} | Iterations: {it}")
            # total_rows = len(pd.concat(pd.read_csv(self.file_path, chunksize=chunksize)))
            # print(f"\nTotal Rows: {total_rows}")
            # end_time = time.time()
            # total_read_time = end_time - start_time
            # print(f"Total Read Time: {total_read_time:.2f} seconds")

            # Base read
            # start_time = time.time()
            self.dataset = pd.read_csv(self.file_path)
            print(self.dataset.shape)
            # end_time = time.time()
            # read_time = end_time - start_time
            # print(read_time)

        except FileNotFoundError:

            print(f"Error: File '{self.file_path}' not found.")
            create_base_dataset()

        except pd.errors.EmptyDataError:

            print(f"Error: File '{self.file_path}' is empty.")

        except pd.errors.ParserError:

            print(f"Error: Parsing error while reading '{self.file_path}'.")

        except Exception as e:  # Catch-all for other unexpected errors

            print(f"Unexpected error while reading '{self.file_path}': {e}")

    def insert_bool(self):
        # for example the firs iteration the win coordinate is 10.246, 15.270
        win_coords = [0.0, 0.0]
        mask1 = self.dataset.iloc[:, 0] == win_coords[0]  # Assuming column names or positions
        mask2 = self.dataset.iloc[:, 1] == win_coords[1]
        self.dataset['matched'] = mask1 & mask2
        self.dataset.head()


if __name__ == '__main__':
    dataset = Dataset()
