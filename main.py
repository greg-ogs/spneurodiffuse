"""
Created on Tuesday, January 22 of 2024 by Greg
"""
import os
import PySpin
# import matplotlib.pyplot as plt
# import time
# from skimage.segmentation import slic
# from skimage.segmentation import mark_boundaries
# from skimage.util import img_as_float
# from skimage import io
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
import keyboard
from PIL import Image as im
import numpy as np
import mysql.connector

from IA import BackPropagation

class Camera:
    def __init__(self):
        self.image = None
        self.continue_recording = True
        mydb = mysql.connector.connect(
            host="localhost",
            user="Greg",
            password="contpass01",
            database="AIRY"
        )

        mycursor = mydb.cursor()

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
            result &= result
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
            # Retrieve and display images
            while (self.continue_recording):
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
                        self.image = np.dstack((C, B))
                        data = im.fromarray(self.image)
                        # self.spixel()
                        # self.center_of_the_beam()
                        winner_class = bp.predict(data)

                        if keyboard.is_pressed('ENTER'):
                            # print('Program is closing...')

                            # Close figure
                            # plt.close('all')
                            # input('Done! Press Enter to exit...')
                            self.continue_recording = False
                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False
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


caminstance = Camera()
caminstance.capture()

