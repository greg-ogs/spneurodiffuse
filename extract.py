import mysql.connector
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class sql_query_e:
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
        df.to_csv('file.csv', index=False)


class plotting_data:
    def __init__(self):
        self.table = pd.read_csv('C:/Users/grego/Downloads/GitHub/1-4.CSV')
        print(self.table)
        sns.set_theme(style="darkgrid")

    def plot_table(self, column_y, title):
        sns.relplot(
            data=self.table,
            x="Step", y=column_y,
            palette="magma",
            hue="Iteration",
            markers=True,
            dashes=False,
            kind="line"
        )
        plt.title(title)
        name = title + '_plot.png'
        plt.savefig(name, dpi=1000)
        plt.show()


# query_e = sql_query_e()
# query_e.save_query()
plot = plotting_data()
plot.plot_table("Physical Memory Load [%]", "Physical Memory Load over Time")
plot.plot_table("CPU Package Power [W]", "CPU Package Power over Time")
plot.plot_table("Total CPU Utility [%]", "Total CPU Utility over Time")