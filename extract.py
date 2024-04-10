import mysql.connector
import numpy as np
import pandas as pd


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


query_e = sql_query_e()
query_e.save_query()
