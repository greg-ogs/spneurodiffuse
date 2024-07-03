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
        s = sns.relplot(
            data=self.table_times,
            x="ID", y='times',
            palette="viridis",
            hue="control",
            # kind="line",
            # units="Iteration", estimator=None
            # markers=True,
            # dashes=False
        )
        s.set_xlabels("Iteration")
        s.set_ylabels("Time (s)")
        title = 'times'
        plt.title(title)
        name = title + '_plot.png'
        plt.savefig(name, dpi=1000)
        plt.show()


# query_e = SqlQueryE()
# query_e.save_query()
plot = PlottingData()
# plot.plot_table("Physical Memory Load [%]", "Physical Memory Load over Time")
# plot.plot_table("CPU Package Power [W]", "CPU Package Power over Time")
# plot.plot_table("Total CPU Utility [%]", "Total CPU Utility over Time")
plot.plot_times()
