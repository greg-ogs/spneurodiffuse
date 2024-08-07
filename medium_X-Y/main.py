"""
Created on Tuesday, January 22 of 2024 by Greg
"""
import time

import matplotlib.pyplot as plt
# import time
import tensorflow as tf
import keyboard
import numpy as np
import PySpin
from PIL import Image as im
import mysql.connector
from IA import BackPropagation


class WinnerMove:
    def BDWL22(self):
        return -0.035, 0.04, True

    def BDWL23(self):
        return -0.035, 0.012, True

    def BDWL32(self):
        return -0.02, 0.06, True

    def BDWL33(self):
        return -0.03, 0.013, True

    def BDWLS(self):
        return -0.035, 0.04, True

    def BDWR22(self):
        return 0.035, 0.04, True

    def BDWR23(self):
        return 0.035, 0.012, True

    def BDWR32(self):
        return 0.02, 0.04, True

    def BDWR33(self):
        return 0.03, 0.013, True

    def BDWRS(self):
        return 0.035, 0.04, True

    def BUPL22(self):
        return -0.035, -0.04, True

    def BUPL23(self):
        return -0.035, -0.012, True

    def BUPL32(self):
        return -0.02, -0.04, True

    def BUPL33(self):
        return -0.03, -0.013, True

    def BUPLS(self):
        return -0.035, -0.04, True

    def BUPR22(self):
        return 0.035, -0.04, True

    def BUPR23(self):
        return 0.035, -0.012, True

    def BUPR32(self):
        return 0.02, -0.04, True

    def BUPR33(self):
        return 0.03, -0.013, True

    def BUPRS(self):
        return 0.035, -0.04, True

    def CDW(self):
        return 0, 0.045, True

    def CENTER(self):
        qry = SqlQuery()
        qry.lab_stop()
        return 0, 0, False

    def CL(self):
        return -0.08, 0, True

    def CR(self):
        return 0.08, 0, True

    def CUP(self):
        return 0, -0.045, True

    def ncup(self):
        return 0, -0.02, True

    def ncr(self):
        return 0.04, 0, True

    def ncl(self):
        return -0.04, 0, True

    def ncdw(self):
        return 0, 0.02, True

    def default(self):
        return 0, 0, False


class Camera:
    def __init__(self):

        self.image = None
        self.continue_recording = True

    def capture(self):
        """
        Example entry point; notice the volume of data that the logging event handler
        prints out on debug despite the fact that very little really happens in this
        example. Because of this, it may be better to have the logger set to lower
        level in order to provide a more concise, focused log.

        :return: True if successful, False otherwise.
        :rtype: bool
        """
        result = True

        # Retrieve singleton reference to system object
        system = PySpin.System.GetInstance()

        # Get current library version
        version = system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        cam_list = system.GetCameras()

        num_cameras = cam_list.GetSize()

        print('Number of cameras detected: %d' % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            cam_list.Clear()
            image_data = [10, 10]

            # Release system instance
            system.ReleaseInstance()

            print('Not enough cameras!')
            # input('Done! Press Enter to exit...')
            return False, image_data

        # Run example on each camera
        for i, cam in enumerate(cam_list):
            print('Running example for camera %d...' % i)
            # time.sleep(2)
            result = self.run_single_camera(cam)
            result &= result
            print('CameraDataSet %d example complete... \n' % i)

        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del cam

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

    def run_single_camera(self, cam):
        """
        This function acts as the body of the example; please see NodeMapInfo example
        for more in-depth comments on setting up cameras.

        :param cam: CameraDataSet to run on.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            result = True

            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Initialize camera
            cam.Init()

            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            # Acquire images
            # result =bool , image_data
            result = self.acquire_images(cam, nodemap, nodemap_tldevice)
            # result &= result
            # Deinitialize camera
            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    def acquire_images(self, cam, nodemap, nodemap_tldevice):
        """
        This function continuously acquires images from a device and display them in a GUI.

        :param cam: CameraDataSet to acquire images from.
        :param nodemap: Device nodemap.
        :param nodemap_tldevice: Transport layer device nodemap.
        :type cam: CameraPtr
        :type nodemap: INodeMap
        :type nodemap_tldevice: INodeMap
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        # global continue_recording #Check uses
        XSetpoint = 640
        YSetpoint = 512
        sNodemap = cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        print('*** IMAGE ACQUISITION ***\n')
        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            cam.BeginAcquisition()

            print('Acquiring images...')

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()
                print('Device serial number retrieved as %s...' % device_serial_number)

            # Close program
            print('Press enter to close the program..')
            bp = BackPropagation()
            qry = SqlQuery()
            # Retrieve and display images
            time1 = time.time()
            aux = 0
            model = tf.keras.models.load_model("model.keras")
            while self.continue_recording:
                aux = aux + 1
                try:

                    #  Retrieve next received image
                    #
                    #  *** NOTES ***
                    #  Capturing an image houses images on the camera buffer. Trying
                    #  to capture an image that does not exist will hang the camera.
                    #
                    #  *** LATER ***
                    #  Once an image from the buffer is saved and/or no longer
                    #  needed, the image must be released in order to keep the
                    #  buffer from filling up.
                    # time.sleep(2)
                    image_result = cam.GetNextImage(100)

                    #  Ensure image completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                    else:

                        # Getting the image data as a numpy array
                        image_data = image_result.GetNDArray()
                        image_data = np.uint8(image_data)
                        # 3D array for superpixel job
                        A = image_data
                        B = image_data
                        C = np.dstack((A, B))
                        image = np.dstack((C, B))
                        data = im.fromarray(image)
                        data = data.resize((375, 300))
                        imname = "img_" + str(aux) + ".png"
                        # imsave = data.save(imname)
                        # plt.imshow(data)
                        # plt.show()
                        winner_class = bp.predict(data, model)
                        switcher = WinnerMove()
                        case = getattr(switcher, winner_class, switcher.default)
                        X, Y, self.continue_recording = case()
                        # qry.map_sql(X, Y)
                        qry.qy(X, Y)
                        qry.next_step()
                        if keyboard.is_pressed('ENTER'):
                            # print('Program is closing...')

                            # Close figure
                            # plt.close('all')
                            # input('Done! Press Enter to exit...')
                            self.continue_recording = False
                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False
            plt.imshow(data)
            plt.show()
            data.save('ERROR.png')
            time2 = time.time()
            print('Time = ' + str(time2 - time1))
            ttime = time2 - time1
            qry.sqltime(ttime)
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

    # def spixel(self):
    #     # deprecated
    #     numSegments = 300  # Segments for superpixels
    #     # apply SLIC and extract the segments
    #     self.segments = slic(self.image, n_segments=numSegments, sigma=5)
    #     # show the output of SLIC
    #     fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.imshow(mark_boundaries(self.image, self.segments))
    #     plt.axis("off")
    #     plt.show()

    # def center_of_the_beam(self):
    #     # Center of the beam code
    #     nlabels = np.amax(self.segments)  # Real segments num
    #     nlabels = nlabels + 1  # Index 0 compensation
    #     nlabels = int(nlabels)
    #     values = []
    #     for i in range(1, nlabels):
    #         coor = np.where(self.segments == i)  # Select the pixels that correspond to segment "i"
    #         # co = [coor[0][0],coor[1][0]]# toma la primera coordenada de cada segmento
    #         # segmentVal = image[co[0]][co[1]][2]#usa la coordenada anterior para buscar el valor en la imagen
    #         arraysize = coor[0].shape  # Num of pixels in this particular segment
    #         arrsiz = arraysize[0]  # tuple to int
    #         meansum = []
    #         for j in range(arrsiz):
    #             coorVal = self.image[coor[0][j]][coor[1][j]][0]  # Value per pixel between 0-255
    #             meansum.append(coorVal)  # Mean value per segment, vector
    #         segmentVal = np.mean(meansum)  # Mean value per segment (segment "i")
    #         values.append(segmentVal)  # Append the mean value for segment number "i" for future comparation
    #     maxsegment = np.where(values == np.amax(values))  # Max mean value in the nlabels segments, return index array
    #     maxS = maxsegment[0] + 1  # Index 0 compensation
    #     maxseg = maxS[0]
    #     maxVC = np.where(
    #         self.segments == maxseg)  # selecciona todas las coordenadas del segmento con valor maximo
    #     # calcular la distancia desde el segmento hasta el centro
    #     arraysz = maxVC[0].shape  # dimencion del conjunto de coordenadas del segmento
    #     arsz = int(arraysz[0] / 2)  # Punto medio del conjunto de coordenadas del segmento
    #     self.XselectCoor = maxVC[1][arsz]  # coordenada intermedia del segmento con mas intencidad en x
    #     self.YselectCoor = maxVC[0][arsz]  # coordenada intermedia del segmento con mas intencidad en y


class SqlQuery:
    def __init__(self):
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="greg",
            password="contpass01",
            database="airy"
        )

        self.mycursor = self.mydb.cursor()

    def sqltime(self, t_time):
        print(type(t_time))
        sql = "INSERT INTO times (ID, time_seg) VALUES (%s, %s)"
        val = (None, t_time)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def map_sql(self, X, Y):
        print("--------------")
        print(X)
        print(Y)
        print("--------------")
        sql = "INSERT INTO MAPING (X, Y) VALUES (%s, %s)"
        val = (X, Y)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def lab_stop(self):
        sql = "UPDATE DATA SET STOP = %s WHERE ID = %s"
        val = (1, 1)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def qy(self, X, Y):
        # def for py
        sql = "UPDATE DATA SET X = %s, Y = %s, SIGNALS = %s WHERE ID = %s"
        val = (X, Y, 1, 1)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

        # Check upload
        self.mycursor.execute("SELECT * FROM DATA WHERE ID = 1")
        myresult = self.mycursor.fetchall()

        list_one = myresult[0]
        x0 = list_one[1]
        y0 = list_one[2]
        print("Upload to  data table...")
        print(x0)
        print(y0)

    def next_step(self):

        # def for py
        while True:
            mydb0 = mysql.connector.connect(
                host="localhost",
                user="greg",
                password="contpass01",
                database="airy"
            )

            mycursor0 = mydb0.cursor()
            mycursor0.execute("SELECT SIGNALS FROM DATA WHERE ID = 1")
            myresult = mycursor0.fetchall()
            myresult = myresult[0]
            myresult = myresult[0]
            mycursor0.close()
            mydb0.close()
            time.sleep(0.01)
            # print(myresult)
            if myresult == 0:
                break


if __name__ == "__main__":
    caminstance = Camera()
    caminstance.capture()
