import mysql.connector
class sql_query:
    def __init__(self):
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="Greg",
            password="contpass01",
            database="AIRY"
        )

        self.mycursor = self.mydb.cursor()

    def qy(self, X, Y):
        # def for py
        sql = "UPDATE AIRY SET X = %s, Y = %s, SIGNALS = %s WHERE ID = %s"
        val = (X, Y, 1, 1)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def next_step(self):
        # def for py
        self.mycursor.execute("SELECT SIGNALS FROM AIRY WHERE ID = 1")
        myresult = self.mycursor.fetchall()
        while myresult == 1:
            self.mycursor.execute("SELECT SIGNALS FROM AIRY WHERE ID = 1")
            myresult = self.mycursor.fetchall()

    def coord(self):
        # def for lv
        self.mycursor.execute("SELECT * FROM DATA WHERE ID = 1")
        myresult = self.mycursor.fetchall()

        list_one = myresult[0]
        x = list_one[1]
        y = list_one[2]
        signal = list_one[3]
        m = [x, y, signal]
        return m

    def signal_0(self):
        # def for lv
        # call before update increments in lv
        # set a timer after lv execute this (4s aprox)
        sql = "UPDATE AIRY SET SIGNALS = %s WHERE ID = %s"
        val = (0, 1)


def get_coord():
    lqy = sql_query()
    lqy.coord()


def get_signal_0():
    lqy = sql_query()
    lqy.signal_0()
