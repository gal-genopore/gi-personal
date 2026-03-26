import argparse
import logging
import numpy as np
import time
from datetime import datetime
import os
import tifffile as tiff
import shutil
import tomllib
import tomli_w
from pathlib import Path

import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QTableWidgetItem, QFileDialog, QMessageBox, QGraphicsProxyWidget
from PyQt6.QtWidgets import QWidget, QSpinBox, QLineEdit, QComboBox, QDoubleSpinBox, QCheckBox, QRadioButton
from PyQt6.QtCore import Qt, QTimer, QRegularExpression, pyqtSignal, pyqtSlot, QThread, QObject
from PyQt6.QtGui import QRegularExpressionValidator, QValidator, QFont
from ExperimentPanelUI import Ui_MainWindow

import pyqtgraph as pg
from pyqtgraph import mkPen

try:
    from pylablib.devices import Newport
except Exception as e:
    print(f"Newport import Error: {e}")

import gxipy as gx

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping, RegenerationMode, Level
from nidaqmx.errors import DaqError

import DBAmpSocket
import socket

import win32event
import queue
from pyAndorSDK2 import atmcd, atmcd_codes
from pyAndorSDK2.atmcd_errors import Error_Codes as iXon_err
import enum

import oxxius

from PM100 import PowerMeterWorker


param_maximumPixelHistory = 400
FAST_KINETICS_FRAME_OVERHEAD = 0.0000172199690714478 # based on 0.3us VS and 16 rows in Solis
FK_ROWS = 16


def convertLogLevelToType(logLevel):
    """
    Converts an integer log level to the corresponding `logging` module's level type.

    Args:
        logLevel (int): The custom integer log level to convert.
                        -2 or less: Corresponds to logging.NOTSET (most verbose)
                        -1: Corresponds to logging.DEBUG
                        0: Corresponds to logging.INFO
                        1: Corresponds to logging.WARNING
                        2: Corresponds to logging.CRITICAL
                        3 or higher: Defaults to logging.ERROR.

    Returns:
        int: The corresponding integer constant from the `logging` module.
    """

    if logLevel <= -2:        
        return logging.NOTSET
    elif logLevel == -1:  
        return logging.DEBUG
    elif logLevel == 0:
        return logging.INFO
    elif logLevel == 1:
        return logging.WARNING
    elif logLevel == 2:
        return logging.CRITICAL
    else:
        # Default for unknown positive integers (e.g., 3, 4, etc.)
        return logging.ERROR # Using ERROR as a safe default for unknown high numbers


class AndorFastKineticAcquisitionWorker(QObject):
    progress = pyqtSignal(int, float, float, float, int, np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, iXon:atmcd, data_queue:queue.Queue):
        super().__init__()
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.iXon = iXon
        self.data_queue = data_queue
        self._abort = False
        self.iXonCodes = atmcd_codes
        self.event_handle = win32event.CreateEvent(None, 0, 0, None)
        

    @pyqtSlot(int, int, float)
    def run_acquisition(self, numFrames, rowHeight, exposureTime):
        TIMEOUT = 1000
        DRV_SUCCESS = 20002
        BURST_ACQ_CMD_TIME = 15*1e-3
        BURST_OVER_TIME = 50*1e-3
        self._abort = False
        totalTimeExpected = numFrames * exposureTime
    
        seriesLength = int(1024 / rowHeight) - 1
        cycleTime = exposureTime + (BURST_ACQ_CMD_TIME + BURST_OVER_TIME) / seriesLength
        # since the time  to command the operation and for the camera to prepare the data is much longer than the actuall exposure
        numBursts = int((totalTimeExpected / cycleTime) / seriesLength)
        pixels_per_burst = 512 * rowHeight * seriesLength
        throttleFactor = max(int(200e-3 / (cycleTime * seriesLength)), 1) # once every 200ms

        # Define Batch Size (e.g., 50 bursts per disk write)
        batchSize = 50
        
        # Pre-allocate the Buffer (Series, Height, Width)
        batchBuffer = np.zeros((batchSize * seriesLength , rowHeight, 512), dtype = np.uint16)
        
        # Configure Camera
        ret = self.iXon.SetIsolatedCropModeEx(Crop.OFF, rowHeight, 512, 1, 1, 1, 512 - rowHeight + 1)
        self.log.debug("Function SetIsolatedCropModeEx returned {} mode = OFF".format(ERROR_CODE[ret]))
        ret = self.iXon.SetSpool(Spool.Disable, self.iXonCodes.Spool_Mode.SPOOL_TO_16_BIT_TIFF, '.', 10)
        self.log.debug("Function SetSpool returned {} Spool = OFF".format(ERROR_CODE[ret]))
        ret = self.iXon.SetAcquisitionMode(self.iXonCodes.Acquisition_Mode.FAST_KINETICS)
        self.log.debug("Function SetAcquisitionMode returned {} mode = Single Scan".format(ret))
        ret = self.iXon.SetFastKineticsEx(exposedRows = rowHeight,
                                                    seriesLength = seriesLength,
                                                    time = exposureTime,
                                                    mode = 4,
                                                    hbin = 1,
                                                    vbin = 1,
                                                    offset = 512 - rowHeight)
        self.log.debug(f'Starting loop for {numBursts} bursts of {seriesLength} frames, height : {rowHeight} rows, exposure time : {exposureTime}s , total pixels : {pixels_per_burst}, throttle : {throttleFactor}, return {ERROR_CODE[ret]}')

        #ret = self.iXon.SetDriverEvent(int(self.event_handle))
        #self.log.debug("Function SetDriverEvent returned {}".format(ret))

        # Start Acquisition (Non-blocking call to driver)
        if self.iXon.StartAcquisition() != DRV_SUCCESS:
            self.error.emit("Failed to start acquisition")
            self.log.warning('Failed to start acquisition')
            self.data_queue.put(None) # Signal writer to close
            self.finished.emit()

            return

        currentBurst = 0
        while currentBurst < numBursts:
            if self._abort:
                self.log.debug('Aborting')
                break

            startEventCall = time.perf_counter()

            #ret = win32event.WaitForSingleObject(self.event_handle, 2000) # win32event.INFINITE)
            #self.log.debug("Function WaitForSingleObject returned {}".format(ret))

            #if ret == 0:
            #    timeoutCounter = TIMEOUT + 1
            #else:
            #    timeoutCounter = TIMEOUT
            
            ret = self.iXon.WaitForAcquisition()
            #self.log.debug("Function WaitForAcquisition returned {}".format(ERROR_CODE[ret]))
            
            if ret == DRV_SUCCESS:
                timeoutCounter = TIMEOUT + 1
            else:
                timeoutCounter = TIMEOUT

            # wait for frames to be ready or timeout
            #timeoutCounter = 0
            #while timeoutCounter < TIMEOUT:
            #    # give some time for the OS/UI
            #    QThread.usleep(100)
            #    #time.sleep(100 * 1e-6) # 100 us
            #    #ret, acc, series = self.iXon.GetAcquisitionProgress()
            #    # check the camera status
            #    ret, status = self.iXon.GetStatus()
            #    if status == Status.Idle: # in idle it means the burst finished
            #        # remember the count for logging
            #        poolIterations = timeoutCounter
            #        # signal the loop to exit
            #        timeoutCounter = TIMEOUT
            
            #    # increase timeout counter
            #    timeoutCounter += 1

            eventCallTime = time.perf_counter() - startEventCall

            if timeoutCounter == TIMEOUT + 1 : # if this wasn't a timeout
                dataReadyTime = time.perf_counter()
                # get the data
                ret, movieBuffer = self.iXon.GetAcquiredData16(pixels_per_burst)

                if ret == DRV_SUCCESS:
                    bufferIdx = (currentBurst % batchSize) * seriesLength
                    # put in our buffer after reshaping 
                    batchBuffer[bufferIdx : bufferIdx + seriesLength] = movieBuffer.reshape(seriesLength, 
                                                                                                rowHeight,
                                                                                                512)
                   
                    # If the batch is full (or this is the last burst), send it to the queue
                    if (currentBurst + 1) % batchSize == 0 or currentBurst + 1 == numBursts:
                        if (currentBurst + 1) % batchSize == 0:
                            actual_fill_end = batchSize * seriesLength
                        else:
                            actual_fill_end = ((currentBurst+1) % batchSize) * seriesLength

                        # We send a COPY of the batch so the worker can't overwrite it 
                        # while the writer is still using it.
                        self.data_queue.put(batchBuffer[:actual_fill_end].copy())
                    dataProcessTime = time.perf_counter() - dataReadyTime

                    # re-trigger the camera
                    startAcqCall = time.perf_counter()
                    if self.iXon.StartAcquisition() != DRV_SUCCESS:
                        self.error.emit(f"Failed to start acquisition in burst {currentBurst}")
                        self.log.warning(f'Failed to start acquisition in burst {currentBurst}')
                        self.data_queue.put(None) # Signal writer to close
                        self.finished.emit()

                        return
                    acqCallTime = time.perf_counter() - startAcqCall
                    
                    # for UI updates we use throttle but also make sure we catch the buffer end case
                    if currentBurst % throttleFactor == 0 or currentBurst + 1 == numBursts or (currentBurst + 1) % batchSize == 0: 
                        self.progress.emit(currentBurst+1, dataProcessTime, eventCallTime, acqCallTime, bufferIdx, batchBuffer[bufferIdx + seriesLength-1])

                    # increase index
                    currentBurst += 1
                else:
                    self.log.warning(f'function GetAcquiredData16 returned {ERROR_CODE[ret]}, burst {currentBurst}')
                    self.error.emit(f'function GetAcquiredData16 returned {ERROR_CODE[ret]}, burst {currentBurst}')
            else:
                self.log.warning(f"Burst {currentBurst} timeout")
                self.error.emit(f"Burst {currentBurst} timeout")
                
        # Clean up
        self.data_queue.put(None) # Signal writer to close
        self.finished.emit()


    def abort(self):
        self._abort = True
        # Tell the hardware to stop immediately
        self.iXon.AbortAcquisition() 
        # Trigger the event in case the thread is stuck in WaitForSingleObject
        self.iXon.CancelWait()
        #win32event.SetEvent(self.event_handle)


    def cleanup(self):
        # Check if handle exists and is actually a PyHANDLE/int
        if self.event_handle:
            try:
                # Explicitly close the Win32 handle
                self.event_handle.Close() 
                #win32api.CloseHandle(self.event_handle)
                self.log.debug("Event handle closed successfully.")
            except Exception as e:
                self.log.error(f"Error closing handle: {e}")
            finally:
                self.event_handle = None


# --- The Consumer (Disk I/O) ---
class TiffWriterWorker(QObject):
    def __init__(self, data_queue:queue.Queue, filename, exposureTime, rowHeight):
        super().__init__()
        self.queue = data_queue
        self.filename = filename
        self.exposureTime = exposureTime
        self.seriesLength = int(1024 / rowHeight) - 1
        self.cycleTime = exposureTime + FAST_KINETICS_FRAME_OVERHEAD 

        self.customTags = [
                            (4876, 'f', 1,          exposureTime , True), # AndorExposureTime 
                            (4878 ,'f', 1,        self.cycleTime , True), # AndorKineticCycleTime
                            (4881 ,'f', 1,        self.cycleTime , True), # AndorAcquisitionCycleTime
                            (297,  'H', 2, (1, self.seriesLength), True)] # PageNumber

    @pyqtSlot()
    def process_queue(self):
        with tiff.TiffWriter(self.filename) as tif: #, bigtiff=True
            while True:
                data = self.queue.get()
                if data is None: 
                    break
                tif.write(data, 
                          contiguous = True, 
                          software = 'Experiment Panel FK',
                          description = 'Bursts of Fast Kinetics', 
                          extratags = self.customTags)

                # since we have enough frames we could froce a disk write so the spool watcher will see it
                tif.filehandle.flush()
                os.fsync(tif.filehandle.fileno())

                self.queue.task_done()
                

class TiffSpoolWatcher(QThread):
    dataBlockReady = pyqtSignal(object)
    errorOccurred = pyqtSignal(str)


    def __init__(self, filePath, pixelCoords, windowSize = 4000):
        super().__init__()
        self.filePath = filePath
        self.x, self.y = pixelCoords
        self.windowSize = windowSize
        self.running = True
        self.log = logging.getLogger(__name__)

        self.log.debug(f"Worker initialized for {filePath} at pixel {pixelCoords} with window of {windowSize}")


    def updatePixelCoordinates(self, pixelCoords):
        self.x, self.y = pixelCoords


    def get3ImagesOffset(self, filePath):
        """
        Gets the offsets in the file to the start of the image data for the first 3 frames.

        Args:
            filePath (str): The path to the TIFF file.

        Returns:
            NDArray : 3 elements , each is the offset of the 1st pixel in that correponding frame
            imageWidth : image width
            imageHeight : image height
            dtype : a dtype of a pixel
        """
        offsets = np.full(3, None)
        imageWidth = -1
        imageHeight = -1
        dtype = None

        # Temporarily silence the tifffile logger
        tf_logger = logging.getLogger('tifffile')
        prev_level = tf_logger.level
        tf_logger.setLevel(logging.CRITICAL)

        try:
            with tiff.TiffFile(filePath) as tif:
                for pageIdx, page in enumerate(tif.pages[:3]): 
                    if pageIdx == 0:
                        imageWidth = page.imagewidth
                        imageHeight = page.imagelength
                        dtype = page.dtype

                    offsets[pageIdx] = page.dataoffsets[0]
        finally:
            # Restore the original logging level
            tf_logger.setLevel(prev_level)

        return offsets, imageWidth, imageHeight, dtype


    def run(self):
        f = None # Initialize to None for the finally block
        offsets = None
        try:
            # --- PHASE 1: BOOTSTRAP ---
            while self.running:
                if os.path.exists(self.filePath):
                    if os.path.getsize(self.filePath) > 1024: # if the file spooling started so the tifffile can do the work
                        # Get initial offsets
                        offsets, imageWidth, imageHeight, dtype = self.get3ImagesOffset(self.filePath)

                        if offsets[2] is not None: # if there were at least 3 pages and we got the data
                            break
                        else:
                            self.log.debug('waiting for 3rd page')
                            self.msleep(20)
                    else:
                        self.log.debug('waiting for file to reach 1024b')
                        self.msleep(20)
                else:
                        self.log.debug('waiting for file')
                        self.msleep(20)

            if offsets is not None and self.running:
                stride = offsets[2] - offsets[1] 
                bytesPerPixel = dtype.itemsize
                pixelIndexInFrame = (self.y * imageWidth) + self.x
                pixelByteStart = pixelIndexInFrame * bytesPerPixel
                
                # Throttling Parameters
                TIME_THRESHOLD = 0.04        # 40ms (25fps) UI refresh limit
                
                f = open(self.filePath, 'rb')
                # we start from the 2nd frame since the first frame has different size IFD
                f.seek(offsets[1])            # Start at first pixel of Frame 1
                
                frameCount = 1
                lastReadTime = time.time()

                while self.running:
                    currentPos = f.tell()
                    fileSize = os.path.getsize(self.filePath)
                    availableBytes = fileSize - currentPos
                    availableFrames = availableBytes // stride
                    
                    elapsedTime = time.time() - lastReadTime

                    # --- THROTTLING LOGIC ---
                    # Read if we have enough frames OR enough time has passed (and there's at least 1 frame)
                    if (availableFrames >= self.windowSize) or (elapsedTime > TIME_THRESHOLD and availableFrames > 0):
                        numFramesToRead = min(availableFrames, self.windowSize)
                        totalBytesToRead = numFramesToRead * stride
                        
                        # read raw bytes
                        blockRaw = f.read(totalBytesToRead)
                        # Reshape to easily target the specific pixel offset in every frame
                        blockArray = np.frombuffer(blockRaw, dtype = np.uint8).reshape(numFramesToRead, stride)
                        
                        # take just the pixel we need from all frames
                        pixelBytes = blockArray[:, pixelByteStart : pixelByteStart + bytesPerPixel]
                        
                        # Extract values into a 1D array
                        pixels = np.frombuffer(pixelBytes.tobytes(), dtype = dtype)

                        if pixels.shape[0] == numFramesToRead: # if we got the number of pixels we intended
                            # Create an array for frame indices
                            indices = np.arange(frameCount, frameCount + numFramesToRead, dtype = np.uint64)
                        else:
                            # create false indices to keep the program running
                            indices = np.arange(1, pixels.shape[0], dtype = np.uint64)
                            self.log.warning(f'Read of {numFramesToRead} frames resulted in {pixels.shape[0]} pixels')
                        
                        # Stack them: shape (N, 2)
                        framesArray = np.column_stack((indices, pixels))
                        
                        self.dataBlockReady.emit(framesArray)
                        frameCount += numFramesToRead
                        lastReadTime = time.time()
                    else:
                        # Sleep slightly to prevent CPU pinning while waiting for frames to accumulate on the disk
                        self.msleep(50)
        except Exception as e:
            self.log.error(f"Throttled Worker Error: {e}", exc_info = True)
            self.errorOccurred.emit(str(e))
        finally:
            if f:
                f.close()
                self.log.debug("File handle closed successfully.")


    def stop(self):
        self.log.debug("Stop signal received by worker.")
        self.running = False


class FanMode(enum.IntEnum):
    Full = 0
    Low = 1
    Off = 2


class OutputAmplificationMode(enum.IntEnum):
    EM = 0
    Conventional = 1


class TTLMode(enum.IntEnum):
    ActiveLow = 0
    ActiveHigh = 1


class ShutterMode(enum.IntEnum):
    Auto = 0
    Open = 1
    Close = 2
    OpenForFVB = 4
    OpenForAny = 5


class BaseClamp(enum.IntEnum):
    Disable = 0
    Enable = 1


class FrameTransferMode(enum.IntEnum):
    OFF = 0
    ON = 1


class Spool(enum.IntEnum):
    Disable = 0
    Enable = 1


class Crop(enum.IntEnum):
    OFF = 0
    ON = 1


class CropMode(enum.IntEnum):
    HighSpeed = 0
    LowLatency = 1
    

class EMGainMode(enum.IntEnum):
    DAC8Bit = 0
    DAQ12Bit = 1
    Linear = 2
    Real = 3


class CoolerMode(enum.IntEnum):
    ReturnsToAmbient = 0
    MaintainTemperature = 1


ERROR_CODE = {
    0: "Unkown",
    20001: "DRV_ERROR_CODES",
    20002: "DRV_SUCCESS",
    20003: "DRV_VXNOTINSTALLED",
    20006: "DRV_ERROR_FILELOAD",
    20007: "DRV_ERROR_VXD_INIT",
    20010: "DRV_ERROR_PAGELOCK",
    20011: "DRV_ERROR_PAGE_UNLOCK",
    20013: "DRV_ERROR_ACK",
    20024: "DRV_NO_NEW_DATA",
    20026: "DRV_SPOOLERROR",
    20034: "DRV_TEMP_OFF",
    20035: "DRV_TEMP_NOT_STABILIZED",
    20036: "DRV_TEMP_STABILIZED",
    20037: "DRV_TEMP_NOT_REACHED",
    20038: "DRV_TEMP_OUT_RANGE",
    20039: "DRV_TEMP_NOT_SUPPORTED",
    20040: "DRV_TEMP_DRIFT",
    20050: "DRV_COF_NOTLOADED",
    20053: "DRV_FLEXERROR",
    20066: "DRV_P1INVALID",
    20067: "DRV_P2INVALID",
    20068: "DRV_P3INVALID",
    20069: "DRV_P4INVALID",
    20070: "DRV_INIERROR",
    20071: "DRV_COERROR",
    20072: "DRV_ACQUIRING",
    20073: "DRV_IDLE",
    20074: "DRV_TEMPCYCLE",
    20075: "DRV_NOT_INITIALIZED",
    20076: "DRV_P5INVALID",
    20077: "DRV_P6INVALID",
    20083: "P7_INVALID",
    20089: "DRV_USBERROR",
    20091: "DRV_NOT_SUPPORTED",
    20095: "DRV_INVALID_TRIGGER_MODE",
    20099: "DRV_BINNING_ERROR",
    20990: "DRV_NOCAMERA",
    20991: "DRV_NOT_SUPPORTED",
    20992: "DRV_NOT_AVAILABLE"
}


class ROIcoordinates (tuple, enum.Enum):
    # Left, Right, Bottom, Top
    CropCenter32x32 = (241,272 ,240              , 271)
    CropCenter64x64 = (219,282 ,224              , 287)
    Full512         = (1  ,512 ,1                , 512)
    Fast16x512      = (1  ,512 ,512 - FK_ROWS + 1, 512)


class Status(enum.IntEnum):
    Idle = iXon_err.DRV_IDLE
    TempCycle = iXon_err.DRV_TEMPCYCLE
    Acquiring = iXon_err.DRV_ACQUIRING
    AccumTimeNotMet = iXon_err.DRV_ACCUM_TIME_NOT_MET
    KineticTimeNotMet = iXon_err.DRV_KINETIC_TIME_NOT_MET
    ErrorAck = iXon_err.DRV_ERROR_ACK
    AcqBuffer = iXon_err.DRV_ACQ_BUFFER
    SpoolError = iXon_err.DRV_SPOOLERROR


class TemperatureStatus(enum.IntEnum):
    Off = iXon_err.DRV_TEMP_OFF
    NotStabilized = iXon_err.DRV_TEMP_NOT_STABILIZED
    Stabilized = iXon_err.DRV_TEMP_STABILIZED
    NotReached = iXon_err.DRV_TEMP_NOT_REACHED
    OutRange = iXon_err.DRV_TEMP_OUT_RANGE
    Drift = iXon_err.DRV_TEMP_DRIFT


class VideoThread(QThread):
    newFrameSignal = pyqtSignal(np.ndarray)
    _instance = None  # Class variable to store instance reference


    def __init__(self, cam):
        super().__init__()

        # remember this instance
        VideoThread._instance = self  # Set the instance reference
        # keep a handle to the camera
        self.cam = cam
        # Register callback
        self.cam.data_stream[0].register_capture_callback(self.onNewFrame)


    @staticmethod
    def onNewFrame(rawImage):
        # get the instance
        instance = VideoThread._instance  # Access the stored instance
        # check if the class was initiated
        if instance is not None:
            if rawImage.get_status() == gx.GxFrameStatusList.INCOMPLETE:
                print("incomplete frame")
            else:
                # Get numpy array from mono raw images
                numpyImage = rawImage.get_numpy_array().T
                # emit it through a signal
                instance.newFrameSignal.emit(numpyImage)
            

class PulseGenerator:
    def __init__(self, deviceName = "Dev1"):
        """
        Initialize the pulse generator for NI 6221
        
        Args:
            deviceName (str): Name of DAQ device (default: "Dev1")
        """
        self.deviceName = deviceName
        self.aoTask = None

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        
    def generatePulseWaveform(self, amplitude, duration_clk, period_clk):
        """
        Generate a single pulse waveform with the pulse at the end of the period
        
        Args:
            amplitude (float): Peak amplitude of the pulse (V)
            duration_clk (int): Duration of the pulse in clock pulses
            period_clk (int): Period of pulse repetition in clock pulses

            
        Returns:
            numpy.ndarray: Generated waveform
        """
        
        # Create one period of the waveform
        waveform = np.zeros(period_clk)
        
        if duration_clk > 0 and duration_clk < period_clk: # if the paramaters are legit
            # set the pulse to the amplitude during the pulse duration in the end of the period
            waveform[-duration_clk:] = amplitude

        waveform = np.clip(waveform, 0.0, 5.0)
            
        return waveform
    
    
    def setupExternalClockPulseGeneration(self, 
                                            aoChannel = "ao0",
                                            clockSourceChannel = "PFI0",
                                            amplitude = 5.0,
                                            duration_clk = 10,
                                            period_clk = 1000):
        """
        Setup analog output pulses with external clock synchronization
        
        Args:
            aoChannel (str): Analog output channel (e.g., "ao0")
            clockSourceChannel (str): External clock source pin (e.g., "PFI0", "PFI1", etc.)
            amplitude (float): Amplitude of pulses (V)
            duration_clk (init): Duration of the pulse in clock pulses
            period_clk (int): Period of pulse repetition in clock pulses
        """
        
        # Generate the pulse waveform
        waveform = self.generatePulseWaveform(amplitude, duration_clk, period_clk)
        
        if self.aoTask is not None: # if a task was already created
            # end the task
            self.aoTask.close()
            self.aoTask = None
        
        # Create analog output task
        self.aoTask = nidaqmx.Task()
        
        # compose the full path to the channel
        aoChannelFull = f"{self.deviceName}/{aoChannel}"
        # remember the channel used
        self.aoChannel = aoChannelFull
        # Add analog output channel
        self.aoTask.ao_channels.add_ao_voltage_chan(aoChannelFull,
                                                    min_val = 0.0,
                                                    max_val = 5.0)
        
        # Configure timing with external clock
        clockSourceFull = f"/{self.deviceName}/{clockSourceChannel}"
        
        self.aoTask.timing.cfg_samp_clk_timing( rate = 1000,               # this assume 1Khz external clock
                                                source = clockSourceFull,  # External clock source
                                                active_edge = Edge.RISING,
                                                sample_mode = AcquisitionType.CONTINUOUS,
                                                samps_per_chan = len(waveform))
        
        # Write waveform to buffer
        self.aoTask.write(waveform, auto_start = False)
        
        self.log.debug( "Setup complete:")
        self.log.debug(f"  Output Channel: {aoChannelFull}")
        self.log.debug(f"  Clock Source: {clockSourceFull}")
        self.log.debug(f"  Amplitude: {amplitude} V")
        self.log.debug(f"  Pulse Duration: {duration_clk} clocks")
        self.log.debug(f"  Period: {period_clk} clocks")
        self.log.debug(f"  Waveform Length: {len(waveform)} samples")

        
    def setupInternalClockPulseGeneration(  self, 
                                            aoChannel = "ao0",
                                            amplitude = 5.0,
                                            duration_ms = 10,
                                            period_ms = 1000):
        """
        Setup analog output pulses with internal clock synchronization
        
        Args:
            aoChannel (str): Analog output channel (e.g., "ao0")
            clockSourceChannel (str): External clock source pin (e.g., "PFI0", "PFI1", etc.)
            amplitude (float): Peak amplitude of pulses (V)
            duration_ms (init): DDuration of the pulse in ms
            period_ms (int): Period of pulse repetition in mss
        """
        
        # Generate the pulse waveform
        waveform = self.generatePulseWaveform(amplitude, duration_ms, period_ms)
        
        if self.aoTask is not None: # if a task was already created
            # end the task
            self.aoTask.close()
            self.aoTask = None

        # Create analog output task
        self.aoTask = nidaqmx.Task()
        
        # compose the full path to the channel
        aoChannelFull = f"{self.deviceName}/{aoChannel}"
        # remember the channel used
        self.aoChannel = aoChannelFull
        # Add analog output channel
        self.aoTask.ao_channels.add_ao_voltage_chan(aoChannelFull,
                                                    min_val = 0.0,
                                                    max_val = 5.0)
        
        # since we give the pulse timing in ms use 1Khz clock
        self.aoTask.timing.cfg_samp_clk_timing(1000, sample_mode = AcquisitionType.CONTINUOUS)

        # Write waveform to buffer
        self.aoTask.write(waveform)
        
        self.log.debug( "Setup complete:")
        self.log.debug(f"  Output Channel: {aoChannelFull}")
        self.log.debug(f"  Amplitude: {amplitude} V")
        self.log.debug(f"  Pulse Duration: {duration_ms} ms")
        self.log.debug(f"  Period: {period_ms} ms")
        self.log.debug(f"  Waveform Length: {len(waveform)} samples")

    
    def startGeneration(self):
        """Start the pulse generation"""
        if self.aoTask is not None:
            self.aoTask.start()
            self.log.debug("Pulse generation started")
        else:
            self.log.warning("Task not configured.")

    
    def stopGeneration(self):
        """Stop the pulse generation"""
        if self.aoTask is not None:
            self.aoTask.stop()
            self.aoTask.close()
            self.aoTask = None
            self.log.debug("Pulse generation stopped.")

            # set the output to 0
            self.aoTask = nidaqmx.Task()
            self.aoTask.ao_channels.add_ao_voltage_chan(self.aoChannel)
            self.aoTask.write(0)
            self.aoTask.stop()
            self.aoTask.close()
            self.aoTask = None
            self.log.debug("Output set to 0.")
    

    def cleanup(self):
        """Clean up resources"""
        if self.aoTask is not None:
            self.aoTask.close()
            self.aoTask = None
            self.log.debug("Resources cleaned up.")


class TriggeredPulseGenerator:
    """
    Generates a single-shot pulse of a specific duration in response to 
    a digital trigger on a PFI line.
    """

    def __init__(self, dev_name: str, trigger_pin: str, output_pin: str, 
                 pulse_duration_us: float, counter_idx: int = 0):
        """
        Args:
            dev_name (str): NI device name (e.g., 'Dev1').
            trigger_pin (str): The PFI pin for the trigger (e.g., 'PFI0').
            output_pin (str): The PFI pin to output the pulse (e.g., 'PFI4').
            pulse_duration_us (float): Duration in microseconds (10 to 500).
            counter_idx (int): The index of the hardware counter to use.
        """
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Convert microseconds to seconds for DAQmx
        duration_sec = pulse_duration_us * 1e-6
        
        self.task = None
        self._is_running = False
        self.counter_path = f"/{dev_name}/ctr{counter_idx}"
        self.trigger_path = f"/{dev_name}/{trigger_pin}"
        self.output_terminal = f"/{dev_name}/{output_pin}"

        try:
            self.task = nidaqmx.Task()

            # 1. Create the Pulse Channel
            # We use a very small 'low_time' for the initial delay (idle state)
            # and set the 'high_time' to desired pulse duration.
            chan = self.task.co_channels.add_co_pulse_chan_time(
                counter = self.counter_path,
                units = nidaqmx.constants.TimeUnits.SECONDS,
                idle_state = Level.LOW,
                initial_delay = 0.0, 
                low_time = 1e-7, # Minimal delay before pulse starts
                high_time = duration_sec
            )
            self.log.debug(f'Created channel for counter {self.counter_path}')

            # 2. Route the output to the physical pin
            chan.co_pulse_term = self.output_terminal
            self.log.debug(f'Routed output to pin {self.output_terminal}')

            # 3. Configure Timing for a Single Shot (Finite)
            self.task.timing.cfg_implicit_timing(
                sample_mode=AcquisitionType.FINITE, 
                samps_per_chan = 1
            )
            self.log.debug('Configures oneshot')

            # 4. Configure the Hardware Start Trigger
            self.task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=self.trigger_path,
                trigger_edge = Edge.RISING
            )
            self.log.debug(f'Configure trigger to {self.trigger_path}')

            # 5. Allow the counter to trigger again without restarting the task
            self.task.triggers.start_trigger.retriggerable = True

            self.log.debug(f"Initialized {pulse_duration_us}us pulse on {output_pin} triggered by {trigger_pin}")

        except nidaqmx.DaqError as e:
            self.log.error(f"Configuration Error: {e}")
            self.close()
            raise


    def start(self):
        if self.task and not self._is_running:
            self.task.start()
            self._is_running = True


    def stop(self):
        if self.task and self._is_running:
            self.task.stop()
            self._is_running = False


    def close(self):
        if self.task:
            self.task.close()
            self.task = None
            self._is_running = False


    def __enter__(self): 
        return self
    
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.close()


class  FlowPressureState(enum.IntEnum):
    OFF = 0
    PUMP = 1
    HOLD = 2
    ADJUST = 3


class FlowPressureController:
    """
    Controls two digital output lines (Flow and Pressure) to manage
    four specific system states.
    """
    
    STATES = {
        FlowPressureState.PUMP  : {"description": "Pump: Flow ON, Pressure ON",   "values": [True , True]},
        FlowPressureState.HOLD  : {"description": "Hold: Flow OFF, Pressure ON",  "values": [False, True]},
        FlowPressureState.ADJUST: {"description": "Adjust: Flow ON, Pressure OFF","values": [True , False]},
        FlowPressureState.OFF   : {"description": "All OFF",                      "values": [False, False]}
    }

    def __init__(self, deviceName: str = 'Dev1', flowLine: str = 'port0/line0', pressureLine: str = 'port0/line1'):
        """
        Args:
            deviceName: NI device name (e.g., 'Dev1')
            flowLine: Digital line for flow (e.g., 'port0/line0')
            pressureLine: Digital line for pressure (e.g., 'port0/line1')
        """
        self.log = logging.getLogger(self.__class__.__name__)
        self.lines = [f"{deviceName}/{flowLine}", f"{deviceName}/{pressureLine}"]

        self.task = None

        try:
            self.task = nidaqmx.Task()
            
            # Add both lines to one task so we can write to them as a list/array
            self.task.do_channels.add_do_chan(
                f"{self.lines[0]},{self.lines[1]}",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            
            self.log.debug(f"Initialized Controller on {self.lines}")
        except Exception as e:
            if self.task is not None:
                self.task.close()
                self.task = None

            self.log.warning(f"Failed to initialized pump controller on {self.lines} error {e}")
        

    def setState(self, stateKey: int):
        if self.task is not None:
            """Updates the DO lines based on the selected state."""
            if stateKey not in self.STATES:
                self.log.error(f"Invalid State: {stateKey}")
                return

            state = self.STATES[stateKey]
            self.log.debug(f"Switching to State {stateKey}: {state['description']}")
            
            # Write the boolean list [Flow, Pressure] to the hardware
            self.task.write(state['values'])


    def close(self):
        if self.task is not None:
            try:
                self.setState(0) # close all valves
                self.task.close() # close the NI task
                self.log.debug("Hardware closed successfully.")
            except Exception as e:
                self.log.error(f"Error during shutdown: {e}")
            finally:
                self.task = None           


    def __enter__(self):
        """Standard entry point for the context manager."""
        return self
    
 
    def __exit__(self, *args): 
        self.close()


class ClockDivder:
    """
    Generates a continuous digital square wave with a 50% duty cycle and its complimentary 
    (inverted) version on two separate Digital Output (DO) lines, all synchronized to an 
    external clock signal.
    
    The pattern is generated based on a given division factor (N), resulting in an output 
    frequency of F_in / N. The pattern is N/2 high samples followed by N/2 low samples.
    """

    @staticmethod
    def generate_division_waveform(divisor: int) -> np.ndarray:
        """
        Generates the 50% duty cycle waveform pattern for the given divisor N.
        
        Args:
            divisor (int): The even integer by which to divide the input frequency (N).

        Raises:
            ValueError: If the divisor is not an even integer greater than 1.
            
        Returns:
            np.ndarray: A NumPy array containing the digital pattern ([True, True, ..., False, False, ...]).
        """
        if divisor <= 1 or divisor % 2 != 0:
            raise ValueError(f"Divisor must be an even integer greater than 1 for 50% duty cycle. Received: {divisor}")
        
        half_period = divisor // 2
        
        # Create N/2 samples of True (High)
        true_part = np.ones(half_period, dtype = np.bool_)
        # Create N/2 samples of False (Low)
        false_part = np.zeros(half_period, dtype = np.bool_)
        
        # Concatenate into one vector
        return np.concatenate((true_part, false_part))


    def __init__(self, dev_name: str, clock_source_pin: str, output_line: str, output_line_inverted: str, divisor: int):
        """
        Initializes the digital output task with external clocking and dynamic waveform generation.

        Args:
            dev_name (str): The name of your NI device (e.g., 'Dev1').
            clock_source_pin (str): The pin receiving the external clock signal (e.g., 'PFI0'). 
                                    This becomes the Sample Clock source.
            output_line (str): The Digital Output line for the NON-INVERTED output (e.g., 'port0/line0').
            output_line_inverted (str): The Digital Output line for the INVERTED output (e.g., 'port0/line1').
            divisor (int): The even integer by which to divide the input frequency (N).
            
        Raises:
            DaqError: If the NI-DAQmx configuration fails.
        """
        
        # create a logger instance
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.dev_name = dev_name
        self.divisor = divisor
        self.task = None
        self._is_running = False
        
        # Generate the standard waveform based on the desired division
        self.waveform_data = self.generate_division_waveform(divisor)
        
        # Generate the inverted waveform (1 - data flips 1s to 0s and vice-versa)
        self.inverted_waveform_data = 1 - self.waveform_data
        
        # Fully qualified pin paths for configuration
        full_clock_pin = f"/{self.dev_name}/{clock_source_pin}"
        full_output_line = f"/{self.dev_name}/{output_line}"
        full_output_line_inverted = f"/{self.dev_name}/{output_line_inverted}"

        self.log.debug(f"Configuring Dual Digital Output Task for F_in / {self.divisor}")
        self.log.debug(f"  External Clock Source: {full_clock_pin}")
        self.log.debug(f"  Primary Output Line (Non-Inverted): {full_output_line}")
        self.log.debug(f"  Secondary Output Line (Inverted): {full_output_line_inverted}")
        self.log.debug(f"  Waveform Pattern: {self.waveform_data}")
        
        try:
            self.task = nidaqmx.Task()

            # --- 1a. CONFIGURE PRIMARY DIGITAL OUTPUT CHANNEL (Non-Inverted) ---
            # This channel is added first, so it corresponds to the first row of the data array.
            self.task.do_channels.add_do_chan(
                full_output_line, 
                line_grouping = LineGrouping.CHAN_FOR_ALL_LINES 
            )
            
            # --- 1b. CONFIGURE SECONDARY DIGITAL OUTPUT CHANNEL (Inverted) ---
            # This channel is added second, so it corresponds to the second row of the data array.
            self.task.do_channels.add_do_chan(
                full_output_line_inverted, 
                line_grouping = LineGrouping.CHAN_FOR_ALL_LINES 
            )
            
            # --- 2. CONFIGURE TIMING (EXTERNAL CLOCK) ---
            # This sets the external signal as the Sample Clock
            self.task.timing.cfg_samp_clk_timing(
                source = full_clock_pin,
                rate = 3000,  # Rate is conceptual in external clock mode
                active_edge = Edge.RISING, 
                sample_mode = AcquisitionType.CONTINUOUS,
                samps_per_chan = len(self.waveform_data) # Set buffer size to the pattern length
            )

            # --- 3. CONFIGURE BUFFER REGENERATION ---
            self.task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            # --- 4. WRITE DATA (2D array for 2 channels) ---
            # Data is written as a 2D array: [ [Channel1 Data], [Channel2 Data] ]
            combined_data = np.array([self.waveform_data, self.inverted_waveform_data], dtype = np.bool_)
            # The digital output will step through this array, one column per external clock tick.
            self.task.write(combined_data, auto_start = False)
            
        except DaqError as e:
            self.log.error(f"DAQmx Configuration Error: {e}")
            self.close()
            raise


    def start(self):
        """
        Starts the pulse generation task.
        """
        if self.task and not self._is_running:
            self.log.debug("Starting synchronous dual digital waveform output.")
            self.task.start()
            self._is_running = True
        elif self._is_running:
            self.log.warning("Digital waveform generation task is already running.")


    def stop(self):
        """
        Stops the pulse generation task.
        """
        if self.task and self._is_running:
            self.log.debug("Stopping digital waveform output.")
            self.task.stop()
            self._is_running = False


    def close(self):
        """
        Clears the DAQmx task and releases resources.
        """
        if self.task:
            if self._is_running:
                self.task.stop()
            self.log.debug("Clearing DAQmx digital waveform generation task resources.")
            self.task.close()
            self.task = None
            self._is_running = False


    def __enter__(self):
        """
        Context manager entry point.
        """
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point, ensures cleanup.
        """
        self.close()


class AppWindow(QMainWindow):
    # Define a signal to start the fast kinetic worker with parameters
    requestRunAcqWorkerFK = pyqtSignal(int, int, float)


    def __init__(self, args):
        super().__init__()

        # start the logger
        self.log = logging.getLogger(__name__)
        logging.basicConfig(level = convertLogLevelToType(args.verbose))

        # use the layout created in QT designer
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.lblFlashOn.setVisible(False)
        self.ui.lblSettingPending.setVisible(False)
        self.ui.sbVoltage_mv.setEnabled(False)  # for now the voltage will be set on the DBAmp application
        self.ui.sbVoltage_mv.setMaximum(5000)
        # add IP validation to the LineEdit input
        self.ipRegex = QRegularExpression(r"^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
        self.ipValidator = QRegularExpressionValidator(self.ipRegex, self)   
        self.ui.leIP.setValidator(self.ipValidator)    
        # to give a realtime feedback to the user connect to the text change event
        self.ui.leIP.textChanged.connect(self.onAddressChange)

        # init variables
        self.closingState = False
        self.setupFileVersion = 'Experiment 1'
        self.msgIdx = 0 # IP message index
        self.socketConnected = False
        self.saveReactionTimeout = 10000
        self.timer1sec = QTimer()
        self.timer100msec = QTimer()
        self.expRunning = False
        self.andorFrameLeft = None
        self.andorFrameRight = None
        self.andorFrameBottom = None
        self.andorFrameTop = None
        self.andorFrameSize = None
        self.andorBin = 1
        self.andorEMCCDGain = -999
        self.andorTemperatureSetpoint = -999
        self.andorExposureTime = -999
        self.actualAndorExpTime = -999
        self.andorCycleTime = -1
        self.setIP(args.host, args.port)
        self.genDummyFeed = args.dummy
        self.networkDrive = args.network_drive
        self.traceFrameIdx = np.zeros(param_maximumPixelHistory, dtype = np.uint64)
        self.tracePixels = np.zeros(param_maximumPixelHistory, dtype = np.uint16)
        self.tracePtr = 0 # Current position in the ring buffer
        self.tiffWatcher = None
        self.combinerBox = None
        self.powerPrecisionFactorL1 = 10
        self.powerPrecisionFactorL2 = 10
        self.powerPrecisionFactorL3 = 10
        self.searchLegNumber = 0
        self.searchCycle = 1
        self.laserBlink = None
        self.data_queue = None
        self.writer_thread = None
        self.writer = None
        self.acq_thread = None
        self.acq_worker = None
        self.frameFK = np.zeros((FK_ROWS, 512))
        self.fastKineticRunning = False
        self.singleFrameAcq = False

        # Create pulse generator instance
        self.pulseGen = PulseGenerator()

        # Rotation matrix for 45 degrees
        theta = np.radians(45)
        self.diagFactor = np.cos(theta)
        self.R = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta), np.cos(theta)]])

        # open a window for the wide field camera view
        self.wideFieldCameraView = WideFieldView(self)
        self.wideFieldCameraView.show()

        # open a window for the Andor camera view
        self.andoraView = AndorView()
        self.andoraView.show()

        # setup the Andor camera
        self.setupAndorCamera()

        # create the object for the stage and get information about it
        self.setupStage()
        # initial mapping of directions to controller and axis
        self.setDefaultStageParameters()

        if self.stage is not None: # if we have a stage
            # read current values of velocity and accl from the stage
            self.velocity, self.accel = self.stage.get_velocity_parameters(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                                                                           addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text())
        
            # update GUI with information from the stage
            self.ui.sbVelocity.setValue(self.velocity)
            self.ui.sbAcceleration.setValue(self.accel)
            for ctrlAddr in self.addresses:
                self.ui.lwAddr.addItem(str(ctrlAddr))
        
            self.updatePosXY()
            self.updatePosZ()
       
        # create a device manager for the wide field camera and set it up
        self.device_manager = gx.DeviceManager()
        self.setupWideFieldCamera()

        # update GUI to match the current setting of the wide field camera
        self.ui.sbExposureTime_us.setValue(self.imgExposureTime)
        ExposureTimeLog = np.log10(self.ui.sbExposureTime_us.value())
        self.ui.dialExposureTime.setValue(int(ExposureTimeLog * 100))
        self.ui.sbGain.setValue(self.imgGain)
        self.ui.sbOffsetX.setValue(self.imgOffsetX)
        self.ui.sbOffsetY.setValue(self.imgOffsetY)
        self.ui.sbWidth.setValue(self.imgWidth)
        self.ui.sbHeight.setValue(self.imgHeight)

        # create a socket for the DBAmp
        self.connectToDBAmp()

        # find available combiners
        self.log.debug("Get list of combiners")
        combinersLister = oxxius.CombinersLister()
        self.combinersList = combinersLister.get_list()

        if len(self.combinersList) > 0: # if there is at least one
            # create a combiner box object
            self.combinerBox = oxxius.Combiner(self.combinersList)

            # open the connection to it
            self.combinerBox.open()

            # get data from the combiner and update the GUI
            self.retrieveCombinerData()

            self.log.debug("Created combiner object")
        else:
            # make a note on the failure in the log
            self.log.warning("No combiner found")
            QMessageBox.warning(self, 'oxxius laser box error', "No laser combiner found")

        # Initialize the worker thread
        wavelength = 595.0 # nm
        self.powermeter = PowerMeterWorker(wavelength)
        self.powermeter.reading_updated.connect(self.onNewPowerReading)
        self.powermeter.error_occurred.connect(self.onPowermeterError)

        # Start the worker thread
        self.powermeter.start()

        # create the pump object
        self.pnumaticPump = FlowPressureController(flowLine = self.ui.leFlowPort.text(), pressureLine = self.ui.lePressurePort.text())

        # connect the actions to the graphical elements
        self.connectActionsGUI()

        # start the timers
        self.timer1sec.timeout.connect(self.on1secTick)
        self.timer1sec.start(1000)
        self.timer100msec.timeout.connect(self.on100msecTick)
        self.timer100msec.start(100)

        self.log.debug('Init ended')
              

    def closeEvent(self, event):
        # make sure other other threads will be aware of the closing state
        self.closingState = True
        # stop the timers and disconnect the handler so they will not call objects we are going to destroy
        self.timer1sec.timeout.disconnect(self.on1secTick)
        self.timer100msec.timeout.disconnect(self.on100msecTick)
        self.timer100msec.stop()
        self.timer1sec.stop()

        # close the child windows
        if self.wideFieldCameraView:
            self.wideFieldCameraView.close() # Close the window.
        if self.andoraView:
            self.andoraView.close() # Close the window.
        # close the devices
        if self.stage is not None:
            self.stage.close()

        if self.cam is not None:
            self.cam.stream_off()
            self.cam.data_stream[0].unregister_capture_callback()
            self.cam.close_device()

        self.pulseGen.stopGeneration()
        self.pulseGen.cleanup()

        self.pnumaticPump.close()

        if self.acq_worker is not None:
            self.acq_worker.abort()

        if self.acq_thread is not None:
            self.acq_thread.quit()
            if not self.acq_thread.wait(2000): # Wait up to 2 seconds
                self.log.warning('Acq thread did not ended nicely')
                self.acq_thread.terminate() # Force kill if it's stuck

        # Flush the data queue and stop the Writer
        if self.data_queue is not None:
            self.data_queue.put(None) # Signal the writer to finish its loop

        if self.writer_thread is not None:
            self.writer_thread.quit()
            if not self.writer_thread.wait(2000): # Wait up to 2 seconds
                self.log.warning('Tiff writer thread did not ended nicely')
                self.writer_thread.terminate() # Force kill if it's stuck

        if self.socketConnected: 
            self.socketDBAmp.close()

        if self.iXon is not None:
            ret = self.iXon.AbortAcquisition()
            self.log.debug("Function AbortAcquisition returned {}".format(ERROR_CODE[ret]))
            ret = self.iXon.SetShutter(TTLMode.ActiveHigh, self.iXonCodes.Shutter_Mode.PERMANENTLY_CLOSED, 10, 10)
            self.log.debug("Function SetShutter returned {}".format(ERROR_CODE[ret]))

            # Clean up
            ret = self.iXon.ShutDown()
            self.log.debug("Function ShutDown returned {}".format(ERROR_CODE[ret]))

        if self.laserBlink is not None:
            self.laserBlink.close()
            self.laserBlink = None

        if len(self.combinersList) > 0: # if there was a combiner
            # disable analog modulation on laser 2
            self.combinerBox.set_analog_modulation(2, False)
            self.combinerBox.close()

        # stop the power meter thread
        self.powermeter.stop()
        self.powermeter.wait() # Wait for the thread to finish

        if self.tiffWatcher and self.tiffWatcher.isRunning(): # if a tiff watcher thread is running
            self.tiffWatcher.stop()
            self.tiffWatcher.wait()

        event.accept()


    def retrieveCombinerData(self):
        """
        Retrieves and updates display elements with data from the connected combiner box.

        This method fetches various operational parameters from the `combinerBox` object,
        such as firmware version, maximum power per laser line, laser type, and
        emission wavelength. It then updates the corresponding UI elements with this
        retrieved information.

        Args:
            None

        Returns:
            None
        """
        # update firmaware
        self.ui.lblLaserCombinerFirmware.setText(f'Firmware: {self.combinerBox.firmware}')

        # get precision factor for GUI
        self.powerPrecisionFactorL1 = 10 ** self.ui.sbPowerL1_mw.decimals()
        self.powerPrecisionFactorL2 = 10 ** self.ui.sbPowerL2_mw.decimals()
        self.powerPrecisionFactorL3 = 10 ** self.ui.sbPowerL3_mw.decimals()

        # update maximum power
        maxPower = int(float(self.combinerBox.max_power(1)))
        self.ui.sldPowerL1.setMaximum(maxPower * self.powerPrecisionFactorL1) # to match the double spin box precision
        self.ui.barPowerL1.setMaximum(maxPower)
        self.ui.sbPowerL1_mw.setMaximum(maxPower)
        self.ui.sbLaserPowerL1_mw.setMaximum(maxPower)

        maxPower = int(float(self.combinerBox.max_power(2)))
        self.ui.sldPowerL2.setMaximum(maxPower * self.powerPrecisionFactorL2) # to match the double spin box precision
        self.ui.barPowerL2.setMaximum(maxPower)
        self.ui.sbPowerL2_mw.setMaximum(maxPower)
        self.ui.sbLaserPowerL2_mw.setMaximum(maxPower)

        maxPower = int(float(self.combinerBox.max_power(3)))
        self.ui.sldPowerL3.setMaximum(maxPower * self.powerPrecisionFactorL3) # to match the double spin box precision
        self.ui.barPowerL3.setMaximum(maxPower)
        self.ui.sbPowerL3_mw.setMaximum(maxPower)
        self.ui.sbLaserPowerL3_mw.setMaximum(maxPower)

        # update laser type and type related values
        laserType = self.combinerBox.laser_type(1)
        self.ui.lblLaserTypeL1.setText(laserType)
        if laserType == 'LCX':
            self.ui.cbEnableL1.setText("Open shutter")
            laserPowerSP = self.combinerBox.power(1)
            self.ui.sbLaserPowerL1_mw.setValue(float(laserPowerSP))
            self.ui.sbLaserPowerL1_mw.setVisible(True)
        else:
            self.ui.sbLaserPowerL1_mw.setVisible(False)

        laserType = self.combinerBox.laser_type(2)
        self.ui.lblLaserTypeL2.setText(laserType)
        if laserType == 'LCX':
            self.ui.cbEnableL2.setText("Open shutter")
            laserPowerSP = self.combinerBox.power(2)
            self.ui.sbLaserPowerL2_mw.setValue(float(laserPowerSP))
            self.ui.sbLaserPowerL2_mw.setVisible(True)
        else:
            self.ui.sbLaserPowerL2_mw.setVisible(False)

        laserType = self.combinerBox.laser_type(3)
        self.ui.lblLaserTypeL3.setText(laserType)
        if laserType == 'LCX':
            self.ui.cbEnableL3.setText("Open shutter")
            laserPowerSP = self.combinerBox.power(3)
            self.ui.sbLaserPowerL3_mw.setValue(float(laserPowerSP))
            self.ui.sbLaserPowerL3_mw.setVisible(True)
        else:
            self.ui.sbLaserPowerL3_mw.setVisible(False)

        # update laser wavelength
        laserWavelength = self.combinerBox.emission_wavelength(1)
        self.ui.lblWavelengthL1.setText(laserWavelength + " nm")

        laserWavelength = self.combinerBox.emission_wavelength(2)
        self.ui.lblWavelengthL2.setText(laserWavelength + " nm")

        laserWavelength = self.combinerBox.emission_wavelength(3)
        self.ui.lblWavelengthL3.setText(laserWavelength + " nm")

        # update initial laser status
        laserEnabled = self.combinerBox.emission_status(1).name == 'ON'
        self.ui.cbEnableL1.setChecked(laserEnabled)
        laserEnabled = self.combinerBox.emission_status(2).name == 'ON'
        self.ui.cbEnableL2.setChecked(laserEnabled)
        laserEnabled = self.combinerBox.emission_status(3).name == 'ON'
        self.ui.cbEnableL3.setChecked(laserEnabled)

        # update initial power setpoint
        laserPowerSP = self.combinerBox.power_setpoint(1)
        self.ui.sbPowerL1_mw.setValue(float(laserPowerSP))
        self.ui.sldPowerL1.setValue(int(self.ui.sbPowerL1_mw.value()) * self.powerPrecisionFactorL1)
        laserPowerSP = self.combinerBox.power_setpoint(2)
        self.ui.sbPowerL2_mw.setValue(float(laserPowerSP))
        self.ui.sldPowerL2.setValue(int(self.ui.sbPowerL2_mw.value()) * self.powerPrecisionFactorL2)
        laserPowerSP = self.combinerBox.power_setpoint(3)
        self.ui.sbPowerL3_mw.setValue(float(laserPowerSP))
        self.ui.sldPowerL3.setValue(int(self.ui.sbPowerL3_mw.value()) * self.powerPrecisionFactorL3)


    def on1secTick(self):
        if self.expRunning: # if we are wating for the experiment to end
            if self.socketConnected and self.ui.cbDBAmpEnable.isChecked(): # if there is connection to the DBAmp
                commOK, self.msgIdx, recStatus = DBAmpSocket.getRecStatDBamp(self.socketDBAmp, self.msgIdx)
                if not recStatus: # if recording ended
                    if self.ui.cbUseVoltage.isChecked() : # if the user want the application to handle voltage commands
                        # turn voltage to zero                    
                        commOK, self.msgIdx = DBAmpSocket.setDrillModeDBamp(self.socketDBAmp, self.msgIdx, drillMode = False, drvVoltage = '0')
                    # turn LED OFF
                    self.stopLED()

                    # turn flow off
                    if self.pnumaticPump is not None and self.ui.rbPumpAutomatic.isChecked():
                        self.pnumaticPump.setState(FlowPressureState.HOLD)

                    # stop the laser blinking (if it was used)
                    if self.laserBlink is not None:
                        self.laserBlink.close()
                        self.laserBlink = None

                    # disable analog modulation on laser 2
                    if self.combinerBox is not None:
                        self.combinerBox.set_analog_modulation(2, False)
                    
                    if self.ui.leIP.text() != '127.0.0.1': # if the DBAmp is not on the same computer
                        self.robustNetworkMove(self.networkSaveFolder, self.saveFolder)

                    if self.tiffWatcher and self.tiffWatcher.isRunning(): # if a tiff watcher trhead was running
                        self.tiffWatcher.stop()
                        self.tiffWatcher.wait()

                    # update status
                    self.expRunning = False
                    self.ui.lblFlashOn.setVisible(False)
                    self.log.info('Experiment ended')
        
        if self.iXon is not None: # if the Andor camera is working
            # update the display
            (ret, temperature) = self.iXon.GetTemperature()
            self.log.debug("Function GetTemperature returned {}".format(ERROR_CODE[ret]))
            self.ui.lblAndorActualTemp.setText(str(temperature) + ' C')

            if not self.ui.rb16x512fk.isChecked(): # if not in Fast Kinetics mode
                (ret, index) = self.iXon.GetTotalNumberImagesAcquired()
                self.log.debug("Function GetTotalNumberImagesAcquired returned {}".format(ERROR_CODE[ret]))
                (ret, status) = self.iXon.GetStatus()
                self.log.debug("Function GetStatus returned {}".format(ERROR_CODE[ret]))
                self.ui.lblAndorStatus.setText('Status: ' + ERROR_CODE[status]+ ' , Frames: ' + str(index))

        if self.combinerBox is not None:
            # combiner data
            self.ui.lblCombinerState.setText(f'State: {self.combinerBox.state.name}')
            self.ui.lblLaserInterlock.setText(f'Interlock: {self.combinerBox.interlock_status.name}')
            self.ui.lblLaserKey.setText(f'Key: {self.combinerBox.emmision_key_status.name}')

            # laser data
            self.ui.lblStatusL1.setText(f'{self.combinerBox.laser_state(1).name}')
            self.ui.lblStatusL2.setText(f'{self.combinerBox.laser_state(2).name}')
            self.ui.lblStatusL3.setText(f'{self.combinerBox.laser_state(3).name}')

            self.ui.barPowerL1.setValue(int(float(self.combinerBox.power(1))))
            self.ui.barPowerL2.setValue(int(float(self.combinerBox.power(2))))
            self.ui.barPowerL3.setValue(int(float(self.combinerBox.power(3))))

            self.ui.lblPowerL1.setText(f'{float(self.combinerBox.power(1))} mW')
            self.ui.lblPowerL2.setText(f'{float(self.combinerBox.power(2))} mW')
            self.ui.lblPowerL3.setText(f'{float(self.combinerBox.power(3))} mW')

        if self.searchLegNumber > 0: # if we are running a search spiral
            legLength = self.ui.sbSpiralGap.value() * self.searchCycle
            if self.searchLegNumber == 1: # if 1st leg
                # move East
                self.log.debug(f'Moving E {legLength}')
                self.moveEW(legLength)
            elif self.searchLegNumber == 2: # if 2nd leg
                # move South
                self.log.debug(f'Moving S {legLength}')
                self.moveSN(legLength)
                # next legs are longer so update to next cycle
                self.searchCycle += 1
            elif self.searchLegNumber == 3: # if 3rd leg
                # move West
                self.log.debug(f'Moving W {legLength}')
                self.moveEW(-legLength)
            elif self.searchLegNumber == 4: # if 4th leg
                # move North
                self.log.debug(f'Moving N {legLength}')
                self.moveSN(-legLength)
                # next legs are longer so update to next cycle
                self.searchCycle += 1

            # set for next leg
            self.searchLegNumber += 1
            if self.searchLegNumber == 5: # if we did 4 legs (turns)
                # repeat from start
                self.searchLegNumber = 1

            if self.searchCycle == 10: # we reached the 10th cycle
                 # make a note on ending the search pattern
                self.log.warning("Reached 10 cycles in search pattern - stopping")
                QMessageBox.warning(self, 'Spiral search', "Reached 10 cycles in search pattern - stopping")
                self.searchLegNumber = 0


    def on100msecTick(self):
        if self.genDummyFeed: # for testing purpose , generate random data
            numpyImage = np.random.triangular(100, 550, 1000, (3000, 2000))
            self.wideFieldCameraView.setNewImage(numpyImage)

            if self.ui.cbAndorLiveView.isChecked():
                self.getFrameBoundaries()
                numpyImage = np.random.triangular(100, 550, 1000, (self.frameHeight // self.andorBin, self.frameWidth // self.andorBin))
                self.andoraView.updateFrame(numpyImage)

        if self.iXon is not None: # if the Andor camera is working
            if self.andorFrameSize is not None:
                if self.fastKineticRunning:
                    # display the frame on the andor window from the FastKintetic thread
                    self.andoraView.updateFrame(self.frameFK)
                else:
                    if ((not self.expRunning) and  self.ui.cbAndorLiveView.isChecked() and self.singleFrameAcq) or (self.expRunning and not self.ui.rb16x512fk.isChecked()):
                        # get the lastest frame
                        (ret, rawFrameBuffer) = self.iXon.GetMostRecentImage16(self.andorFrameSize)
                        self.log.debug("Function GetMostRecentImage16 returned {} size {}".format(ERROR_CODE[ret],self.andorFrameSize))

                        if ret == self.iXon_success : # if it was a successfull operation
                            # convert the data to the correct shape
                            frame = rawFrameBuffer.reshape(self.frameHeight // self.andorBin, self.frameWidth // self.andorBin)
                            # display the frame on the andor window
                            self.andoraView.updateFrame(frame)

                            self.singleFrameAcq = False


            # get current status
            (ret, status) = self.iXon.GetStatus()
            self.log.debug("Function GetStatus returned {} Status = {}".format(ERROR_CODE[ret],ERROR_CODE[status]))
            self.log.debug('Running is  {}'.format( self.expRunning))
            if  (not self.expRunning) and (status == Status.Idle) and self.ui.cbAndorLiveView.isChecked() : # if we just want to see the live image and there is no acq in progress
                ret = self.iXon.SetAcquisitionMode(self.iXonCodes.Acquisition_Mode.KINETICS)
                self.log.debug("Function SetAcquisitionMode returned {} mode = Kinetics".format(ERROR_CODE[ret]))

                # get a single new frame
                ret = self.iXon.SetNumberKinetics(1)
                self.log.debug("Function SetNumberKinetics returned {}".format(ERROR_CODE[ret]))

                ret = self.iXon.SetSpool(Spool.Disable, self.iXonCodes.Spool_Mode.SPOOL_TO_16_BIT_TIFF, '.', 10)
                self.log.debug("Function SetSpool returned {} Spool = OFF".format(ERROR_CODE[ret]))

                # get the coordinates of the ROI based on user selection
                self.getFrameBoundaries()

                # set view port based on user selection
                ret = self.iXon.SetIsolatedCropModeEx(Crop.ON,self.frameHeight, self.frameWidth, self.andorBin, self.andorBin, self.andorFrameLeft, self.andorFrameBottom)
                self.log.debug("Function SetIsolatedCropModeEx returned {} mode = ON".format(ERROR_CODE[ret]))
                ret = self.iXon.SetIsolatedCropModeType(CropMode.HighSpeed)
                self.log.debug("Function SetIsolatedCropModeType returned {} mode = high Speed".format(ERROR_CODE[ret]))

                self.setupAndorCameraByUserSetting()

                ret = self.iXon.StartAcquisition()
                self.log.debug("Function StartAcquisition returned {} ".format(ERROR_CODE[ret]))

                self.singleFrameAcq = True

            if  self.expRunning : # if there is a running experiment
                if self.andoraView and self.ui.cbAndorPixelTrace.isChecked() and self.tiffWatcher: # if the Andor view window exists and the user want pixel trace
                    # update the tiff watcher with the coordinates
                    self.tiffWatcher.updatePixelCoordinates(self.andoraView.crossLocation)


    def onSingleShotEnd(self):
        self.stopLED()

        # turn flow off
        if self.pnumaticPump is not None and self.ui.rbPumpAutomatic.isChecked():
            self.pnumaticPump.setState(FlowPressureState.HOLD)

        if self.laserBlink is not None:
            self.laserBlink.close()
            self.laserBlink = None

        # disable analog modulation on laser 2
        if self.combinerBox is not None:
            self.combinerBox.set_analog_modulation(2, False)

        # stop the tiff watcher
        if self.tiffWatcher and self.tiffWatcher.isRunning():
            self.tiffWatcher.stop()
            self.tiffWatcher.wait()

        self.expRunning = False
        self.ui.lblFlashOn.setVisible(False)
        self.log.info('Experiment ended')


    def setIP(self, host, port):
        # set the port number
        self.PORT = port
        
        # run the IP validator
        pos = 0
        state, _ , _ = self.ipValidator.validate(host, pos)
        if state == QValidator.State.Acceptable: # if it is a valid IP address
            
            self.host = host
        else:
            self.log.warning(f'Bad IP address {host}')
            QMessageBox.warning(self,'Bad IP address','Defaulting to 127.0.0.1')

            self.host = '127.0.0.1'

        # update the GUI
        self.ui.leIP.setText(self.host)


    def getFrameBoundaries(self):
        if self.ui.rb32.isChecked():
            self.andorFrameLeft, self.andorFrameRight, self.andorFrameBottom, self.andorFrameTop = ROIcoordinates.CropCenter32x32
        elif self.ui.rb64.isChecked():
            self.andorFrameLeft, self.andorFrameRight, self.andorFrameBottom, self.andorFrameTop = ROIcoordinates.CropCenter64x64
        elif self.ui.rb512.isChecked():
            self.andorFrameLeft, self.andorFrameRight, self.andorFrameBottom, self.andorFrameTop = ROIcoordinates.Full512
        elif self.ui.rb16x512fk.isChecked():
            self.andorFrameLeft, self.andorFrameRight, self.andorFrameBottom, self.andorFrameTop = ROIcoordinates.Fast16x512
        else:
            self.log.warning('No crppoing selected for Andor')

        self.frameWidth = self.andorFrameRight - self.andorFrameLeft + 1
        self.frameHeight = self.andorFrameTop - self.andorFrameBottom + 1

        if self.ui.cbAndorBinning.currentText() == '1 x 1 bin':
            self.andorBin = 1
        else:
            self.andorBin = 2

        self.andorFrameSize = (self.frameWidth // self.andorBin) * (self.frameHeight // self.andorBin)


    def setupStage(self):
        try:
            self.stage = Newport.Picomotor8742(multiaddr = True, scan = True)
            self.stageID = self.stage.get_id()
            self.stageInfo = self.stage.get_device_info()
            self.addresses, _ = self.stage.get_addr_map()
            self.stage.autodetect_motors()
            # show information about the stage
            self.ui.lblStageDevice.setText(str(self.stageID))
        except Exception:
            QMessageBox.warning(self, 'Stage error', "No connection to stage")
            self.log.warning("No connection to stage")
            self.stage = None
            self.stageID = 1
            self.stageInfo = 'Dummy'
            self.addresses = []

   
    def setDefaultStageParameters(self):
        self.rowZ1 = 0
        self.rowZ2 = 1
        self.rowZ3 = 2
        self.rowW = 3
        self.rowNW = 4
        self.rowNE = 5
        self.colAddr = 0
        self.colAxis = 1
        self.ui.twMapping.setItem(self.rowZ1, self.colAddr, QTableWidgetItem('1'))
        self.ui.twMapping.setItem(self.rowZ1, self.colAxis, QTableWidgetItem('1'))
        self.ui.twMapping.setItem(self.rowZ2, self.colAddr, QTableWidgetItem('1'))
        self.ui.twMapping.setItem(self.rowZ2, self.colAxis, QTableWidgetItem('2'))
        self.ui.twMapping.setItem(self.rowZ3, self.colAddr, QTableWidgetItem('1'))
        self.ui.twMapping.setItem(self.rowZ3, self.colAxis, QTableWidgetItem('3'))
        self.ui.twMapping.setItem(self.rowW, self.colAddr, QTableWidgetItem('2'))
        self.ui.twMapping.setItem(self.rowW, self.colAxis, QTableWidgetItem('1'))
        self.ui.twMapping.setItem(self.rowNW, self.colAddr, QTableWidgetItem('2'))
        self.ui.twMapping.setItem(self.rowNW, self.colAxis, QTableWidgetItem('2'))
        self.ui.twMapping.setItem(self.rowNE, self.colAddr, QTableWidgetItem('2'))
        self.ui.twMapping.setItem(self.rowNE, self.colAxis, QTableWidgetItem('3'))
        self.ui.twMapping.resizeColumnsToContents()


    def connectActionsGUI(self):
        self.ui.btnS.clicked.connect(self.sClicked)
        self.ui.btnN.clicked.connect(self.nClicked)
        self.ui.btnW.clicked.connect(self.wClicked)
        self.ui.btnE.clicked.connect(self.eClicked)
        self.ui.btnUp.clicked.connect(self.upClicked)
        self.ui.btnDown.clicked.connect(self.downClicked)
        self.ui.btnZeroXY.clicked.connect(self.zeroXYClicked)
        self.ui.btnZeroZ.clicked.connect(self.zeroZClicked)
        self.ui.btnHomeXY.clicked.connect(self.homeXYXClicked)
        self.ui.btnHomeZ.clicked.connect(self.homeZClicked)
        self.ui.sbExposureTime_us.valueChanged.connect(self.onExposureTimeSpinBoxFisnihed)
        self.ui.sbGain.valueChanged.connect(self.cameraSetingChanged)
        self.ui.sbOffsetX.valueChanged.connect(self.cameraSetingChanged)
        self.ui.sbOffsetY.valueChanged.connect(self.cameraSetingChanged)
        self.ui.sbWidth.valueChanged.connect(self.cameraSetingChanged)
        self.ui.sbHeight.valueChanged.connect(self.cameraSetingChanged)
        self.ui.pbfullViewWideField.clicked.connect(self.fullViewWideField)
        self.ui.pbGraphicalROI.clicked.connect(self.onGraphicalROIClicked)
        self.ui.pbStartExp.clicked.connect(self.onStartExpClicked)
        self.ui.actionLoad_setup.triggered.connect(self.loadSetup)
        self.ui.actionSave_setup.triggered.connect(self.saveSetup)
        self.ui.dialExposureTime.sliderReleased.connect(self.onExposureTimeFinished)  # Triggered when user releases
        self.ui.btnSelectDataPath.clicked.connect(self.onSelecDataPathClicked)
        self.ui.btnConnectSocket.clicked.connect(self.onConnectSocketClicked)
        self.ui.cbUseVoltage.toggled.connect(self.onUseVoltageToggled)
        self.ui.sldAndorEmGain.valueChanged.connect(self.onSliderAndorEmGainChange)
        self.ui.sbAndorEmGain.valueChanged.connect(self.onAndorEmGainChange)
        self.ui.sbAndorExposure_ms.valueChanged.connect(self.OnAndorExpTimeChange)
        self.ui.sbAndorTemp_degC.valueChanged.connect(self.onAndorTempChange)
        self.ui.cbAndorBinning.currentTextChanged.connect(self.onAndorBinChnage)
        self.ui.rb32.toggled.connect(self.onAndorRoiToggled)
        self.ui.rb64.toggled.connect(self.onAndorRoiToggled)
        self.ui.rb512.toggled.connect(self.onAndorRoiToggled)
        self.ui.rb16x512fk.toggled.connect(self.onAndorRoiToggled)
        self.ui.btnAbort.clicked.connect(self.onAbortClicked)
        self.ui.rbInternalTiming.toggled.connect(self.onInternalTimingToggle)
        self.ui.sbPulseDuration.valueChanged.connect(self.onPulseDurationChange)
        self.ui.sbPowerL1_mw.valueChanged.connect(self.onPowerSpinBoxFisnihedL1)
        self.ui.sbPowerL2_mw.valueChanged.connect(self.onPowerSpinBoxFisnihedL2)
        self.ui.sbPowerL3_mw.valueChanged.connect(self.onPowerSpinBoxFisnihedL3)

        self.ui.sldPowerL1.valueChanged.connect(self.onPowerSliderChangedL1)
        self.ui.sldPowerL2.valueChanged.connect(self.onPowerSliderChangedL2)
        self.ui.sldPowerL3.valueChanged.connect(self.onPowerSliderChangedL3)

        self.ui.cbEnableL1.toggled.connect(self.onEnableToggleL1)
        self.ui.cbEnableL2.toggled.connect(self.onEnableToggleL2)
        self.ui.cbEnableL3.toggled.connect(self.onEnableToggleL3)

        self.ui.sbLaserPowerL1_mw.valueChanged.connect(self.onLaserPowerSpinBoxFisnihedL1)
        self.ui.sbLaserPowerL2_mw.valueChanged.connect(self.onLaserPowerSpinBoxFisnihedL2)
        self.ui.sbLaserPowerL3_mw.valueChanged.connect(self.onLaserPowerSpinBoxFisnihedL3)

        self.ui.btnStartSearch.clicked.connect(self.onStartSearchClicked)
        self.ui.btnStopSearch.clicked.connect(self.onStopSearchClicked)

        self.ui.rbAlternatingBlink.toggled.connect(self.onLaserBlinkModeToggled)

        self.ui.rbPumpOff.toggled.connect(self.onPumpModeToggled)
        self.ui.rbPumpAdjust.toggled.connect(self.onPumpModeToggled)
        self.ui.rbPumpHold.toggled.connect(self.onPumpModeToggled)
        self.ui.rbPumpFlow.toggled.connect(self.onPumpModeToggled)
        self.ui.rbPumpAutomatic.toggled.connect(self.onPumpModeToggled)

        self.ui.leFlowPort.editingFinished.connect(self.onPumpLineChanged)
        self.ui.lePressurePort.editingFinished.connect(self.onPumpLineChanged)


    def setupWideFieldCamera(self):
        # let the program continue with null camera if nonoe willbe found
        self.cam = None
        # set the defualt setting of the camera
        self.imgExposureTime = self.ui.sbExposureTime_us.value()
        self.imgGain = self.ui.sbGain.value()
        self.imgOffsetX = self.ui.sbOffsetX.value()
        self.imgOffsetY = self.ui.sbOffsetY.value()
        self.imgWidth = self.ui.sbWidth.value()
        self.imgHeight = self.ui.sbHeight.value()

        # find cameras 
        dev_num, dev_info_list = self.device_manager.update_device_list()
        if dev_num == 0: # if there are no cameras
            QMessageBox.warning(self, 'Wide field camera error', "Number of enumerated devices is 0")
            self.log.warning("Wide field camera error - Number of enumerated devices is 0")
        else:
            try:
                # open the first device
                self.cam = self.device_manager.open_device_by_index(1)
                # display vendor id and model
                self.ui.lblDeviceVendor.setText(self.cam.DeviceVendorName.get())
                self.ui.lblDeviceModelName.setText(self.cam.DeviceModelName.get())
                # create the video capture thread
                self.thread = VideoThread(self.cam)
                # connect its signal to the updateImage slot
                self.thread.newFrameSignal.connect(self.onNewFrame)
                # start the thread
                self.thread.start()
                
                # set continuous acquisition at maximum of 25 FPS
                self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
                self.cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
                self.cam.AcquisitionFrameRate.set(25.0)
                # update the GUI limits based on the camera
                self.updateSpinBoxRange(self.cam.ExposureTime.get_range(),self.ui.sbExposureTime_us)
                self.updateSpinBoxRange(self.cam.Gain.get_range(),self.ui.sbGain)
                self.updateWideFieldCameraSettingLimits()
                # get the current setting of the camera
                self.imgExposureTime = self.cam.ExposureTime.get()
                self.imgGain = self.cam.Gain.get()
                self.imgOffsetX = self.cam.OffsetX.get()
                self.imgOffsetY = self.cam.OffsetY.get()
                self.imgWidth = self.cam.Width.get()
                self.imgHeight = self.cam.Height.get()
                # use ROI
                if self.cam.RegionMode.get()[0] == 0:
                    self.cam.RegionMode.set(gx.GxSwitchEntry.ON)
                # start data acquisition
                self.cam.stream_on()
            except Exception as e:
                QMessageBox.warning(self, 'Wide field camera error', "Failed to access camera")

                self.log.warning(f"An error occurred: {e}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                line_number = exc_traceback.tb_lineno
                self.log.warning(f"An error of type {exc_type.__name__} occurred:")
                self.log.warning(f"Error message: {exc_value}")
                self.log.warning(f"Line number: {line_number}")


    def setupAndorCameraByUserSetting(self):
        if self.iXon is not None:
            if self.andorTemperatureSetpoint != self.ui.sbAndorTemp_degC.value(): # if there was a change since last time
                self.andorTemperatureSetpoint = self.ui.sbAndorTemp_degC.value()
                ret = self.iXon.SetTemperature(self.ui.sbAndorTemp_degC.value())
                self.log.debug("Function SetTemperature returned {} Setpoint = {}".format(ERROR_CODE[ret],self.ui.sbAndorTemp_degC.value()))

            if self.andorEMCCDGain != self.ui.sbAndorEmGain.value(): # if there was a change since last time
                self.andorEMCCDGain = self.ui.sbAndorEmGain.value()
                ret = self.iXon.SetEMCCDGain(self.ui.sbAndorEmGain.value())
                self.log.debug("Function SetEMCCDGain returned {}".format(ERROR_CODE[ret]))

            (ret, actualGain) = self.iXon.GetEMCCDGain()
            self.log.debug("Function GetEMCCDGain returned {} Gain = {}".format(ERROR_CODE[ret],actualGain))

            if self.andorExposureTime != self.ui.sbAndorExposure_ms.value(): # if there was a change since last time
                self.andorExposureTime = self.ui.sbAndorExposure_ms.value()
                ret = self.iXon.SetExposureTime(self.ui.sbAndorExposure_ms.value() * 1e-3) # the field is in ms
                self.log.debug("Function SetExposureTime returned {}".format(ERROR_CODE[ret]))

            (ret , expTime, accumTime, cycleTime) = self.iXon.GetAcquisitionTimings()
            self.log.debug("Function GetAcquisitionTimings returned {} Exposure time = {}s, Accumulation time = {}s, Cycle time = {}s".format(ERROR_CODE[ret],expTime,accumTime,cycleTime))

            self.actualAndorExpTime = expTime
            self.andorCycleTime = cycleTime
            self.ui.lblAndorActualExposure.setText(str(round(expTime*1000,2)) + ' ms / ' + str(round(cycleTime*1000,4)) + ' ms')
            self.ui.lblSettingPending.setVisible(False)            


    def insertAndorPropertyToTable(self, andorProperty, value):
        lastRow = self.ui.twAndorSetting.rowCount()
        self.ui.twAndorSetting.insertRow(lastRow)
        self.ui.twAndorSetting.setItem(lastRow, 0, QTableWidgetItem(andorProperty))
        self.ui.twAndorSetting.setItem(lastRow, 1, QTableWidgetItem(value))


    def setupAndorCamera(self):
        self.iXon = atmcd()  # Load the atmcd library
        self.iXonCodes = atmcd_codes
        self.iXon_success = iXon_err.DRV_SUCCESS

        ret = self.iXon.Initialize("")  # Initialize camera
        if ret == self.iXon_success: # if the initialization succeeded
            (ret, eprom, coffile, vxdrev, vxdver, dllrev, dllver) = self.iXon.GetSoftwareVersion()
            self.log.debug("Function GetSoftwareVersion returned {} Versions: {}.{}.{}.{}.{}.{}".format(ERROR_CODE[ret], eprom, coffile, vxdrev, vxdver, dllrev, dllver))
            self.insertAndorPropertyToTable('Andor eprom.cof.vxdrev.vxdver.dllrev.dllver version', "{}.{}.{}.{}.{}.{}".format(eprom, coffile, vxdrev, vxdver, dllrev, dllver))

            (ret, iSerialNumber) = self.iXon.GetCameraSerialNumber()
            self.log.debug("Function GetCameraSerialNumber returned {} Serial No: {}".format(ERROR_CODE[ret], iSerialNumber))
            self.insertAndorPropertyToTable('Camera S/N', str(iSerialNumber))

            (ret, headModel) = self.iXon.GetHeadModel()
            self.log.debug("Function GetHeadModel returned {} Head model: {}".format(ERROR_CODE[ret], headModel.decode('utf-8', errors = 'ignore')))
            self.insertAndorPropertyToTable('Head model', headModel.decode('utf-8', errors = 'ignore'))

            ret = self.iXon.SetFanMode(FanMode.Full) 
            self.log.debug("Function SetFanMode returned {}".format(ERROR_CODE[ret]))
            self.insertAndorPropertyToTable('Fan mode', 'Full')

            ret = self.iXon.CoolerON()
            self.log.debug("Function CoolerON returned {}".format(ERROR_CODE[ret]))
            self.insertAndorPropertyToTable('Cooler', 'On')

            ret = self.iXon.SetCoolerMode(CoolerMode.MaintainTemperature)
            self.log.debug("Function SetCoolerMode returned {} setting mode to MaintainTemperature".format(ERROR_CODE[ret]))

            (ret, adcCount) = self.iXon.GetNumberADChannels()
            self.log.debug("Function GetNumberADChannels returned {} Number of ADC = {}".format(ERROR_CODE[ret],adcCount))
            self.insertAndorPropertyToTable('Number of ADC', str(adcCount))

            ( ret, ampCount) = self.iXon.GetNumberAmp()
            self.log.debug("Function GetNumberAmp returned {} Number of amplifiers = {}".format(ERROR_CODE[ret],ampCount))
            self.insertAndorPropertyToTable('Number of amplifiers', str(ampCount))

            ret = self.iXon.SetOutputAmplifier(OutputAmplificationMode.EM)
            self.log.debug("Function SetOutputAmplifier returned {}".format(ERROR_CODE[ret]))

            (ret, ampDesc) = self.iXon.GetAmpDesc(OutputAmplificationMode.EM,21)
            self.log.debug("Function GetAmpDesc returned {}, Amp = {}".format(ERROR_CODE[ret],ampDesc))
            self.insertAndorPropertyToTable('Amplifier', ampDesc)

            ret = self.iXon.SetEMGainMode(EMGainMode.Real)
            self.log.debug("Function SetEMGainMode returned {}, Amp mode = Real EM gain".format(ERROR_CODE[ret]))
            self.insertAndorPropertyToTable('Amplifier mode', 'Real EM gain')

            ret, minGain, maxGain = self.iXon.GetEMGainRange()
            self.log.debug("Function GetEMGainRange returned {} min = {}, max = {}".format(ERROR_CODE[ret],minGain,maxGain))
            self.insertAndorPropertyToTable('EM gain range', str(minGain) + ' - ' +str(maxGain))
            self.ui.sldAndorEmGain.setMinimum(minGain)
            self.ui.sldAndorEmGain.setMaximum(maxGain)

            ret = self.iXon.SetHSSpeed(OutputAmplificationMode.EM, 0) # fastest speed
            self.log.debug("Function SetHSSpeed returned {}".format(ERROR_CODE[ret]))
            ( ret, hsSpeedMHz) = self.iXon.GetHSSpeed(0,OutputAmplificationMode.EM,0)
            self.log.debug("Function GetHSSpeed returned {} HS Speed is {}Mhz".format(ERROR_CODE[ret], hsSpeedMHz))
            self.insertAndorPropertyToTable('HS Speed', str(hsSpeedMHz) + ' Mhz')

            ret = self.iXon.SetPreAmpGain(2)
            self.log.debug("Function SetPreAmpGain returned {}".format(ERROR_CODE[ret]))
            ( ret, preAmpGainDesc) = self.iXon.GetPreAmpGainText(2, 21)
            self.log.debug("Function GetPreAmpGainText returned {} preAmpGain is {}".format(ERROR_CODE[ret], preAmpGainDesc.value))
            # convert the array of chars to string
            arr = np.char.asarray(preAmpGainDesc, unicode = "utc=8")
            preAmpGainDescStr = ""
            preAmpGainDescStr = preAmpGainDescStr.join(arr)
            self.insertAndorPropertyToTable('Pre amp gain', preAmpGainDescStr)
  
            ret = self.iXon.SetVSSpeed(0)
            self.log.debug("Function SetVSSpeed returned {}".format(ERROR_CODE[ret]))
            ( ret, vsSpeedMicrosecond) = self.iXon.GetVSSpeed(0)
            self.log.debug("Function GetVSSpeed returned {} VS Speed is {}us".format(ERROR_CODE[ret], vsSpeedMicrosecond))
            self.insertAndorPropertyToTable('VS Speed', str(vsSpeedMicrosecond) + ' us')

            ret , vsAmplitudesCount = self.iXon.GetNumberVSAmplitudes()
            self.log.debug("Function GetNumberVSAmplitudes returned {} Number of avilabe amplitudes = {}".format(ERROR_CODE[ret], vsAmplitudesCount))
            self.insertAndorPropertyToTable('VS Amplitude avilable', str(vsAmplitudesCount))

            ret = self.iXon.SetVSAmplitude(3)
            ret, GetVSAmplitudeCString3 = self.iXon.GetVSAmplitudeString(3)
            self.log.debug("Function GetVSAmplitudeString returned {}".format(ERROR_CODE[ret]))
            # convert the array of chars to string
            arr = np.char.asarray(GetVSAmplitudeCString3, unicode = "utc=8")
            GetVSAmplitudeString3 = ""
            GetVSAmplitudeString3 = GetVSAmplitudeString3.join(arr)
            self.log.debug("Function SetVSAmplitude returned {} Set = {}".format(ERROR_CODE[ret],GetVSAmplitudeString3))
            self.insertAndorPropertyToTable('VS Amplitude selected', GetVSAmplitudeString3)

            ret = self.iXon.SetBaselineClamp(BaseClamp.Enable)
            self.log.debug("Function SetBaselineClamp returned {}".format(ERROR_CODE[ret]))
            (ret, baseClamp) = self.iXon.GetBaselineClamp()
            self.log.debug("Function GetBaselineClamp returned {} Clamp = {}".format(ERROR_CODE[ret], baseClamp == 1))
            if baseClamp == 1:
                self.insertAndorPropertyToTable('Baseline clampd', 'ON')
            else:
                self.insertAndorPropertyToTable('Baseline clampd', 'OFF')

            (ret, xpixels, ypixels) = self.iXon.GetDetector()
            self.log.debug("Function GetDetector returned {} xpixels = {} ypixels = {}".format(ERROR_CODE[ret], xpixels, ypixels))
            self.insertAndorPropertyToTable('Detector xpixels', str(xpixels))
            self.insertAndorPropertyToTable('Detector ypixels', str(ypixels))

            ret = self.iXon.SetFrameTransferMode(FrameTransferMode.ON)
            self.log.debug("Function SetFrameTransferMode returned {} mode = ON".format(ERROR_CODE[ret]))
            self.insertAndorPropertyToTable('Frame transfer', 'ON')

            ret = self.iXon.SetReadMode(self.iXonCodes.Read_Mode.IMAGE)
            self.log.debug("Function SetReadMode returned {} mode = Image".format(ERROR_CODE[ret]))
            self.insertAndorPropertyToTable('Read mode', 'IMAGE')

            self.insertAndorPropertyToTable('Isolated Crop Mode Type', 'High speed')

            ret = self.iXon.SetTriggerMode(self.iXonCodes.Trigger_Mode.INTERNAL)
            self.log.debug("Function SetTriggerMode returned {} mode = Internal".format(ERROR_CODE[ret]))
            self.insertAndorPropertyToTable('Trigger mode', 'Internal')

            ret = self.iXon.SetShutter(TTLMode.ActiveHigh,self.iXonCodes.Shutter_Mode.PERMANENTLY_OPEN, 0, 0)
            self.log.debug("Function SetShutter returned {}".format(ERROR_CODE[ret]))
            self.insertAndorPropertyToTable('Shutter mode', 'PERMANENTLY_OPEN')

            self.setupAndorCameraByUserSetting()
        else:
            QMessageBox.warning(self,'Andor camera issue','Failed to Initialize Andor camera')
            self.log.warning('Failed to Initialize Andor camera')
            self.ui.lblAndorStatus.setText('Camera failed')
            self.ui.lblAndorActualTemp.setText('--C')
            self.ui.lblAndorActualExposure.setText('-- ms')
            try:
                self.iXon.ShutDown()
            finally:
                self.iXon = None


    @pyqtSlot(np.ndarray)
    def onNewFrame(self, numpyImage):
        if not self.closingState: # if we are not in the middle of clsing windows 
            if numpyImage is not None : # if there is a new frame
                # display the new framer
                self.wideFieldCameraView.setNewImage(numpyImage)


    @pyqtSlot(float)
    def onNewPowerReading(self, reading):
        """
        Updates the GUI with the new power reading.
        """
        self.ui.lblPowermeterMeasPower.setText(f"Power: {reading * 1000:.3f} mW")
        self.ui.lblPowermeterResource.setText(f'Resource {self.powermeter.resource_name}')
        self.ui.lblPowermeterModel.setText(f'Model : {self.powermeter.modelName}')
        self.ui.lblPowermeterSerialNumber.setText(f'S/N :{self.powermeter.serialNumber}')


    @pyqtSlot(str)
    def onPowermeterError(self, message):
        self.log.warning(f"Power meter error: {message}")
        QMessageBox.critical(self, "Error of power meter", message)
        self.powermeter.stop()


    @pyqtSlot(object)
    def onNewTraceBlock(self, traceBlock):
        if self.andoraView and self.ui.cbAndorPixelTrace.isChecked():
            # traceBlock[:, 0] is frames, traceBlock[:, 1] is pixels
            n_new = traceBlock.shape[0]

            # If the new block is larger than the entire window, just take the tail
            if n_new >= param_maximumPixelHistory:
                self.log.debug(f'Got {n_new} frames')
                self.traceFrameIdx[:] = traceBlock[-param_maximumPixelHistory:, 0]
                self.tracePixels[:] = traceBlock[-param_maximumPixelHistory:, 1]
                self.tracePtr = 0
            else:
                self.log.debug(f'Got {n_new} frames out of {param_maximumPixelHistory} to display')
                # Calculate indices for circular insertion
                indices = (np.arange(self.tracePtr, self.tracePtr + n_new)) % param_maximumPixelHistory
                
                # Vectorized assignment using NumPy advanced indexing
                self.traceFrameIdx[indices] = traceBlock[:, 0]
                self.tracePixels[indices] = traceBlock[:, 1]
                
                # Update pointer for next time
                self.tracePtr = (self.tracePtr + n_new) % param_maximumPixelHistory

            # We still need to align the data for the plot so it looks linear but this is done without a COPY of the data
            #pixelTraceLineFrameIds = np.concatenate((self.traceFrameIdx[self.tracePtr:], self.traceFrameIdx[:self.tracePtr]))
            pixelTraceLine = np.concatenate((self.tracePixels[self.tracePtr:], self.tracePixels[:self.tracePtr]))

            # altough we have the frame numbers , we send just the trace so there will be no need to shift the X axis
            self.andoraView.updatePixelPlot(pixelTraceLine)
            

    @pyqtSlot(str)
    def onTiffWatcherError(self, msg):
        self.log.error(f"GUI received error from Tiff worker: {msg}")


    def onAddressChange(self, text):
        # on a change we can indicate to the user if it failes the validation
        pos = 0
        # run the validator
        state, _, _ = self.ipValidator.validate(text, pos)

        # according to the state change the background color
        if state == QValidator.State.Acceptable:
            self.ui.leIP.setStyleSheet("")  # Reset style if valid
        elif state == QValidator.State.Intermediate:
            self.ui.leIP.setStyleSheet("QLineEdit { background-color: lightyellow; }")  # Partial valid, warn
        else:
            self.ui.leIP.setStyleSheet("QLineEdit { background-color: red; }")  # Invalid, error


    def connectToDBAmp(self):
        if self.socketConnected: # if already connected
            self.socketDBAmp.close()
        # initial state
        self.socketConnected = False
        self.ui.lblStatusDBAmp.setText('OFFLINE')
        self.ui.lblStatusDBAmp.setStyleSheet("color: red;")
        try: # try to connect using the GUI parameters
            self.socketDBAmp = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            self.socketDBAmp.settimeout(5)
            self.socketDBAmp.connect((self.ui.leIP.text(), self.ui.sbPort.value()))
            self.socketConnected = True
            # tell the DBAmp application to connect to the amp
            commOK, self.msgIdx = DBAmpSocket.connectDBamp(self.socketDBAmp, self.msgIdx)
            if commOK:
                self.ui.lblStatusDBAmp.setText('READY')
                self.ui.lblStatusDBAmp.setStyleSheet("color: green;")
            else:
                QMessageBox.warning(self,'Warning','DBAmp application failed to connect to amplifier')
                self.log.warning('DBAmp application failed to connect to amplifier')
        except Exception:
            QMessageBox.warning(self,'Bad socket','Could not connect to the DBAmp application')
            self.log.warning('Could not connect to the DBAmp application')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            line_number = exc_traceback.tb_lineno
            self.log.warning(f"An error of type {exc_type.__name__} occurred:")
            self.log.warning(f"Error message: {exc_value}")
            self.log.warning(f"Line number: {line_number}")            


    def onConnectSocketClicked(self):
        self.connectToDBAmp()


    def updatePosXY(self):
        # get the actual position from the controller
        self.posW = self.stage.get_position(axis = self.ui.twMapping.item(self.rowW, self.colAxis).text(), 
                                            addr = self.ui.twMapping.item(self.rowW, self.colAddr).text())
        self.posNE = self.stage.get_position(axis = self.ui.twMapping.item(self.rowNE, self.colAxis).text(), 
                                             addr = self.ui.twMapping.item(self.rowNE, self.colAddr).text())
        self.posNW = self.stage.get_position(axis = self.ui.twMapping.item(self.rowNW, self.colAxis).text(), 
                                             addr = self.ui.twMapping.item(self.rowNW, self.colAddr).text())
        # rotate the cooridnate systems
        self.posX, self.posY = self.R @ np.array([self.posNW, self.posNE])  # Matrix multiplication
        self.posX = self.posX + self.posW
        # convert to um
        posXum = self.posX / self.ui.sbScaleWE.value()
        posYum = self.posY / self.ui.sbScaleNS.value()
        # update the display
        self.ui.lblPosX.setText(str(round(self.posX)) + '=' + str(round(posXum,2)) + 'um')
        self.ui.lblPosY.setText(str(round(self.posY)) + '=' + str(round(posYum,2)) + 'um')


    def updatePosZ(self):
        # get the actual position from the controller
        posZ = self.stage.get_position(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                                       addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text())
        # convert to um
        posZum = posZ / self.ui.sbScaleZ.value()
        # update the display
        self.ui.lblPosZ.setText(str(round(posZ)) + '=' + str(round(posZum, 2)) + 'um')


    def moveSN(self, steps):
        stepsDiag = int(steps * self.diagFactor)
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowNW, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowNW, self.colAddr).text(),
                           steps = stepsDiag)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowNW, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowNW, self.colAddr).text())
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowNE, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowNE, self.colAddr).text(),
                           steps = stepsDiag)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowNE, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowNE, self.colAddr).text())
        self.updatePosXY()


    def sClicked(self):
        self.moveSN(self.ui.sbStepXY.value())


    def nClicked(self):
        self.moveSN(-self.ui.sbStepXY.value())


    def moveEW(self, steps):
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowW, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowW, self.colAddr).text(),
                           steps = steps)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowW, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowW, self.colAddr).text())
        self.updatePosXY()


    def eClicked(self):
        self.moveEW(self.ui.sbStepXY.value())


    def wClicked(self):
        self.moveEW(-self.ui.sbStepXY.value())


    def upClicked(self):
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text(),
                           steps = self.ui.sbStepZ.value())
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text())
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowZ2, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ2, self.colAddr).text(),
                           steps = self.ui.sbStepZ.value())
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ2, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ2, self.colAddr).text())
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowZ3, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ3, self.colAddr).text(),
                           steps = self.ui.sbStepZ.value())
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ3, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ3, self.colAddr).text())
        self.updatePosZ()


    def downClicked(self):
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text(),
                           steps = -self.ui.sbStepZ.value())
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text())
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowZ2, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ2, self.colAddr).text(),
                           steps = -self.ui.sbStepZ.value())
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ2, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ2, self.colAddr).text())
        self.stage.move_by(axis = self.ui.twMapping.item(self.rowZ3, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ3, self.colAddr).text(),
                           steps = -self.ui.sbStepZ.value())
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ3, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ3, self.colAddr).text())
        self.updatePosZ()
        

    def zeroZClicked(self):
        self.stage.set_position_reference(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                                          addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text())
        self.stage.set_position_reference(axis = self.ui.twMapping.item(self.rowZ2, self.colAxis).text(), 
                                          addr = self.ui.twMapping.item(self.rowZ2, self.colAddr).text())
        self.stage.set_position_reference(axis = self.ui.twMapping.item(self.rowZ3, self.colAxis).text(), 
                                          addr = self.ui.twMapping.item(self.rowZ3, self.colAddr).text())
        self.updatePosZ()


    def zeroXYClicked(self):
        self.stage.set_position_reference(axis = self.ui.twMapping.item(self.rowW, self.colAxis).text(), 
                                          addr = self.ui.twMapping.item(self.rowW, self.colAddr).text())
        self.stage.set_position_reference(axis = self.ui.twMapping.item(self.rowNW, self.colAxis).text(), 
                                          addr = self.ui.twMapping.item(self.rowNW, self.colAddr).text())
        self.stage.set_position_reference(axis = self.ui.twMapping.item(self.rowNE, self.colAxis).text(), 
                                          addr = self.ui.twMapping.item(self.rowNE, self.colAddr).text())
        self.updatePosXY()


    def homeXYXClicked(self):
        self.stage.move_to(axis = self.ui.twMapping.item(self.rowNW, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowNW, self.colAddr).text(),
                           position = 0)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowNW, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowNW, self.colAddr).text())
        self.stage.move_to(axis = self.ui.twMapping.item(self.rowNE, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowNE, self.colAddr).text(),
                           position = 0)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowNE, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowNE, self.colAddr).text())
        self.stage.move_to(axis = self.ui.twMapping.item(self.rowW, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowW, self.colAddr).text(),
                           position = 0)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowW, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowW, self.colAddr).text())
        self.updatePosXY()


    def homeZClicked(self):
        self.stage.move_to(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text(),
                           position = 0)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ1, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ1, self.colAddr).text())
        self.stage.move_to(axis = self.ui.twMapping.item(self.rowZ2, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ2, self.colAddr).text(),
                           position = 0)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ2, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ2, self.colAddr).text())
        self.stage.move_to(axis = self.ui.twMapping.item(self.rowZ3, self.colAxis).text(), 
                           addr = self.ui.twMapping.item(self.rowZ3, self.colAddr).text(),
                           position = 0)
        self.stage.wait_move(axis = self.ui.twMapping.item(self.rowZ3, self.colAxis).text(), 
                             addr = self.ui.twMapping.item(self.rowZ3, self.colAddr).text())
        self.updatePosZ()


    def onExposureTimeFinished(self):
        # the dial range is the log of the ExposureTime in fixed point of 2 decimal places
        ExposureTime = 10 ** (self.ui.dialExposureTime.value() / 100)
        # update the spin box (that will cause an event to update the camera as well)
        self.ui.sbExposureTime_us.setValue(ExposureTime)


    def onExposureTimeSpinBoxFisnihed(self):
        # update also the dial
        ExposureTimeLog = np.log10(self.ui.sbExposureTime_us.value())
        self.ui.dialExposureTime.setValue(int(ExposureTimeLog * 100))
        # update the camera
        self.cameraSetingChanged()


    def roundToStep(self, spinBox):
        spinBox.setValue(int(spinBox.value() / spinBox.singleStep()) * spinBox.singleStep())


    def roundToSpinboxStep(self, val, spinBox):
        return (int(val / spinBox.singleStep()) * spinBox.singleStep())


    def updateSpinBoxRange(self, range, spinBox):
        spinBox.setMinimum(range['min'])
        spinBox.setMaximum(range['max'])
        inc = range['inc']
        if inc == 0:
            spinBox.setSingleStep(1)
        else :
            spinBox.setSingleStep(range['inc'])
            if isinstance(spinBox, QDoubleSpinBox):
                precision = int(np.max(-np.log10(range['inc']), 0))
                spinBox.decimals = precision


    def updateWideFieldCameraSettingLimits(self):
        self.updateSpinBoxRange(self.cam.Width.get_range(), self.ui.sbWidth)
        self.updateSpinBoxRange(self.cam.Height.get_range(), self.ui.sbHeight)
        self.updateSpinBoxRange(self.cam.OffsetX.get_range(), self.ui.sbOffsetX)
        self.updateSpinBoxRange(self.cam.OffsetY.get_range(), self.ui.sbOffsetY)  


    @pyqtSlot()
    def cameraSetingChanged(self):
        # make sure we have legit values
        self.roundToStep(self.ui.sbWidth)
        self.roundToStep(self.ui.sbHeight)
        self.roundToStep(self.ui.sbOffsetX)
        self.roundToStep(self.ui.sbOffsetY)
        # update the internal variables
        self.imgExposureTime = self.ui.sbExposureTime_us.value()
        self.imgGain = self.ui.sbGain.value()
        self.imgOffsetX = self.ui.sbOffsetX.value()
        self.imgOffsetY = self.ui.sbOffsetY.value()
        self.imgWidth = self.ui.sbWidth.value()
        self.imgHeight = self.ui.sbHeight.value()
        # update the camera
        self.updateWideFieldCameraSetting()


    def updateWideFieldCameraSetting(self):
        if self.cam: # if there is a camera object
            # stop data acquisition
            self.cam.stream_off()
            # set exposure
            self.cam.ExposureTime.set(self.imgExposureTime)
            # set gain
            self.cam.Gain.set(self.imgGain)
            # set ROI
            self.cam.Width.set(self.imgWidth)
            self.cam.Height.set(self.imgHeight)
            self.cam.OffsetX.set(self.imgOffsetX)
            self.cam.OffsetY.set(self.imgOffsetY)
            self.updateWideFieldCameraSettingLimits()
            # re-start data acquisition
            self.cam.stream_on()


    def fullViewWideField(self):
        self.ui.sbOffsetX.setValue(0)
        self.ui.sbOffsetY.setValue(0)
        self.ui.sbWidth.setValue(self.cam.SensorWidth.get())
        self.ui.sbHeight.setValue(self.cam.SensorHeight.get())
        self.cameraSetingChanged()


    def roiSelected(self, offsetX,offsetY,width,height):
        # set the GUI for the user selected region
        self.ui.sbWidth.setValue(width)
        self.ui.sbHeight.setValue(height)
        self.ui.sbOffsetX.setValue(offsetX+self.imgOffsetX)
        self.ui.sbOffsetY.setValue(offsetY+self.imgOffsetY)
        # treat this as a new user selected values
        self.cameraSetingChanged()


    def onGraphicalROIClicked(self):
        if self.ui.pbGraphicalROI.isChecked():
            self.wideFieldCameraView.addGraphicalROI()
        else:
            roiRect = self.wideFieldCameraView.getGraphicalROI()
            if roiRect: # if we got anything back
                # set the GUI for the user selected region
                self.ui.sbWidth.setValue(int(roiRect.width()))
                self.ui.sbHeight.setValue(int(roiRect.height()))
                self.ui.sbOffsetX.setValue(int(roiRect.x()) + self.imgOffsetX)
                self.ui.sbOffsetY.setValue(int(roiRect.y()) + self.imgOffsetY)
                # treat this as a new user selected values
                self.cameraSetingChanged()
            # remove the ROI from the plot
            self.wideFieldCameraView.removeGraphicalROI()     


    def loadConfig(self, filePath):
        try:
            with open(filePath, "rb") as f:
                config = tomllib.load(f)

            return config
        except FileNotFoundError:
            return {}
        

    def loadConfigWithDefaults(self, filePath):
        # Define defaults
        defaults = {
            "Wide Field camera": {
                self.ui.sbExposureTime_us.objectName() : 100000,
                self.ui.sbGain.objectName() : 24,
                self.ui.sbOffsetX.objectName() : 0,
                self.ui.sbOffsetY.objectName() : 0,
                self.ui.sbWidth.objectName() : 5496,
                self.ui.sbHeight.objectName() : 3672,
            },
            "LED" : {
                self.ui.sbPulseDuration.objectName() : 10,
                self.ui.sbPulsePeriod.objectName() : 10000,
                self.ui.sbPulseVoltage.objectName() : 5.0,
                self.ui.rbReflectionLED.objectName() : False,
                self.ui.rbInternalTiming.objectName() : False,
            },
            "DBAmp" : {
                self.ui.leIP.objectName() : "127.0.0.1",
                self.ui.sbPort.objectName() : 10285,
                self.ui.sbVoltage_mv.objectName() : 100,
                self.ui.cbDBAmpEnable.objectName() : True,
                self.ui.cbUseVoltage.objectName() : False,
            },
            "Laser" : {
                self.ui.cbEnableL1.objectName() : False,
                self.ui.sbPowerL1_mw.objectName() : 0,
                self.ui.sbLaserPowerL1_mw.objectName() : 0,
                self.ui.cbEnableL2.objectName() : False,
                self.ui.sbPowerL2_mw.objectName() : 0,
                self.ui.sbLaserPowerL2_mw.objectName() : 0,
                self.ui.cbEnableL3.objectName() : False,
                self.ui.sbPowerL3_mw.objectName() : False,
                self.ui.sbLaserPowerL3_mw.objectName() : False,
            },
            "Laser blink" : {
                self.ui.cbBlinkLaserL2.objectName() : False,
                self.ui.leOutputPin.objectName() : "PFI8",
                self.ui.leInvertedOutputPin.objectName() : "port2/line1",
                self.ui.sbClockDivisor.objectName() : 2,
                self.ui.sbLaserPulseDuration_us.objectName() : 100,
                self.ui.rbAlternatingBlink.objectName() : False,
            },
            "OptoSplit" : {
                self.ui.cbGreenChannel.objectName() : True,
                self.ui.cbRedChannel.objectName() : True,
            },
            "Andor" : {
                self.ui.sbAndorExposure_ms.objectName() : 1.0,
                self.ui.sbAndorEmGain.objectName() : 30,
                self.ui.sbAndorTemp_degC.objectName() : -70,
                self.ui.rb32.objectName() : True,
                self.ui.rb64.objectName() : False,
                self.ui.rb512.objectName() : False,
                self.ui.rb16x512fk.objectName() : False,
                self.ui.cbAndorBinning.objectName() : "1 x 1 bin",
            },
            "Pump" : {
                self.ui.leFlowPort : 'port0/line6',
                self.ui.lePressurePort : 'port0/line7',
                self.ui.rbPumpOff.objectName() : True,
                self.ui.rbPumpAdjust.objectName() : False,
                self.ui.rbPumpHold.objectName() : False,
                self.ui.rbPumpFlow.objectName() : False,
                self.ui.rbPumpAutomatic.objectName() : False,
            },
        }

        # Load the user's file
        fileConfig = self.loadConfig(filePath)

        # Deep Merge
        # This fills in the blanks if the user deleted a line
        finalConfig = defaults | fileConfig 
        
        # For nested dictionaries, a simple merge might not be enough, 
        # so we often do this for specific sections:
        #final_config["acquisition"] = defaults["acquisition"] | user_config.get("acquisition", {})
        
        return finalConfig


    def applyConfigToUI(self, config):
        """
        Automatically populates UI widgets based on TOML keys.
        Works for nested dictionaries (sections).
        """
        # Flatten the nested dictionary (e.g., config['Andor']['exposure'])
        # into a single level for easier matching
        flatData = {}
        for section, values in config.items():
            if isinstance(values, dict):
                flatData.update(values)
            else:
                flatData[section] = values

        # Iterate over the keys
        for key, value in flatData.items():
            # Find the widget by its objectName
            widget = self.findChild(QWidget, key)
            
            if widget: # if there is a widget with that name
                # use the appropriate method to assign the value 
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, (QCheckBox, QRadioButton)):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)

        # Handle the stage table data separately
        stageParameters = config.get('Stage', {})

        # Define the keys (e.g., ['Z1', 'Z2', 'X1'])
        targetKeys = stageParameters.keys()
        
        # Iterate through every row currently in the table
        for row in range(self.ui.twMapping.rowCount()):
            # Get the text from the vertical header (the row label)
            headerItem = self.ui.twMapping.verticalHeaderItem(row)
            if headerItem is None:
                continue
                
            headerText = headerItem.text() # This would be "Z1"
            
            # If the header matches a key in our dictionary, fill that row
            if headerText in targetKeys:
                self.ui.twMapping.setItem(row, self.colAddr, QTableWidgetItem(str(stageParameters[headerText][self.colAddr])))
                self.ui.twMapping.setItem(row, self.colAxis, QTableWidgetItem(str(stageParameters[headerText][self.colAxis])))
       
        self.ui.twMapping.resizeColumnsToContents()


    def loadSetup(self):
        fileDialog = QFileDialog(self)
        fileDialog.setWindowTitle("Load File")
        fileDialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        fileDialog.setViewMode(QFileDialog.ViewMode.Detail)
        fileDialog.setNameFilters(["Setup (*.toml)","All file (*)"])

        if fileDialog.exec():
            setupFileName = fileDialog.selectedFiles()[0]
            
            # Load from file
            config = self.loadConfigWithDefaults(setupFileName)
            
            if config:
                # Update the UI widgets automatically
                self.applyConfigToUI(config)

                if self.cam: # if the camera is connected
                    self.updateWideFieldCameraSettingLimits()
                    self.updateWideFieldCameraSetting()

                # attempt to connect with the new setup
                self.connectToDBAmp()
            else:
                QMessageBox.warning(self,'Bad file','Could not load configuration')
                self.log.warning('The setup file is not compatible with current version')
                

    def decodeItem(self, item:QTableWidgetItem):
        if item:
            value = item.text().strip()
        else:
            value = ""
        
        # Basic type conversion for numbers
        try:
            if "." in value :
                value = float(value)
            else: 
                value = int(value)
        except ValueError:
            pass

        return value

    def saveMetadata(self, filename):
        tomlFileName = str(Path(filename).with_suffix(".toml"))
        # Gather variables
        configData = {
            "Version" : {
                "Metadata" : self.setupFileVersion
            },
            "Commnet" : {
                "Commnet" : self.ui.leComment.text()
            },
            "Wide Field camera": {
                self.ui.sbExposureTime_us.objectName() : self.ui.sbExposureTime_us.value(),
                self.ui.sbGain.objectName() : self.ui.sbGain.value(),
                self.ui.sbOffsetX.objectName() : self.ui.sbOffsetX.value(),
                self.ui.sbOffsetY.objectName() : self.ui.sbOffsetY.value(),
                self.ui.sbWidth.objectName() : self.ui.sbWidth.value(),
                self.ui.sbHeight.objectName() : self.ui.sbHeight.value(),
            },
            "LED" : {
                self.ui.sbPulseDuration.objectName() : self.ui.sbPulseDuration.value(),
                self.ui.sbPulsePeriod.objectName() : self.ui.sbPulsePeriod.value(),
                self.ui.sbPulseVoltage.objectName() : self.ui.sbPulseVoltage.value(),
                self.ui.rbReflectionLED.objectName() : self.ui.rbReflectionLED.isChecked(),
                self.ui.rbTransmissionLED.objectName() : self.ui.rbTransmissionLED.isChecked(),
                self.ui.rbInternalTiming.objectName() : self.ui.rbInternalTiming.isChecked(),
                self.ui.rbSyncFrameTiming.objectName() : self.ui.rbSyncFrameTiming.isChecked(),
            },
            "DBAmp" : {
                self.ui.leIP.objectName() : self.ui.leIP.text(),
                self.ui.sbPort.objectName() : self.ui.sbPort.value(),
                self.ui.sbVoltage_mv.objectName() : self.ui.sbVoltage_mv.value(),
                self.ui.cbDBAmpEnable.objectName() : self.ui.cbDBAmpEnable.isChecked(),
                self.ui.cbUseVoltage.objectName() : self.ui.cbUseVoltage.isChecked(),
            },
            "Laser" : {
                self.ui.lblLaserCombinerFirmware.objectName() : self.ui.lblLaserCombinerFirmware.text(),
                "Laser 1 max power [mW]" : self.ui.sbPowerL1_mw.maximum(),
                self.ui.lblLaserTypeL1.objectName() : self.ui.lblLaserTypeL1.text(),
                self.ui.lblWavelengthL1.objectName() : self.ui.lblWavelengthL1.text(),
                self.ui.cbEnableL1.objectName() : self.ui.cbEnableL1.isChecked(),
                self.ui.sbPowerL1_mw.objectName() : self.ui.sbPowerL1_mw.value(),
                self.ui.sbLaserPowerL1_mw.objectName() : self.ui.sbLaserPowerL1_mw.value(),
                "Laser 2 max power [mW]" : self.ui.sbPowerL2_mw.maximum(),
                self.ui.lblLaserTypeL2.objectName() : self.ui.lblLaserTypeL2.text(),
                self.ui.lblWavelengthL2.objectName() : self.ui.lblWavelengthL2.text(),
                self.ui.cbEnableL2.objectName() : self.ui.cbEnableL2.isChecked(),
                self.ui.sbPowerL2_mw.objectName() : self.ui.sbPowerL2_mw.value(),
                self.ui.sbLaserPowerL2_mw.objectName() : self.ui.sbLaserPowerL2_mw.value(),
                "Laser 3 max power [mW]" : self.ui.sbPowerL3_mw.maximum(),
                self.ui.lblLaserTypeL3.objectName() : self.ui.lblLaserTypeL3.text(),
                self.ui.lblWavelengthL3.objectName() : self.ui.lblWavelengthL3.text(),
                self.ui.cbEnableL3.objectName() : self.ui.cbEnableL3.isChecked(),
                self.ui.sbPowerL3_mw.objectName() : self.ui.sbPowerL3_mw.value(),
                self.ui.sbLaserPowerL3_mw.objectName() : self.ui.sbLaserPowerL3_mw.value(),
            },
            "Laser blink" : {
                self.ui.cbBlinkLaserL2.objectName() : self.ui.cbBlinkLaserL2.isChecked(),
                self.ui.leOutputPin.objectName() : self.ui.leOutputPin.text(),
                self.ui.leInvertedOutputPin.objectName() : self.ui.leInvertedOutputPin.text(),
                self.ui.sbClockDivisor.objectName() : self.ui.sbClockDivisor.value(),
                self.ui.sbLaserPulseDuration_us.objectName() : self.ui.sbLaserPulseDuration_us.value(),
                self.ui.rbAlternatingBlink.objectName() : self.ui.rbAlternatingBlink.isChecked(),
                self.ui.rbShortBlink.objectName() : self.ui.rbShortBlink.isChecked(),
            },
            "OptoSplit" : {
                self.ui.cbGreenChannel.objectName() : self.ui.cbGreenChannel.isChecked(),
                self.ui.cbRedChannel.objectName() : self.ui.cbRedChannel.isChecked(),
            },
            "Power meter" : {
                "Resource" : self.powermeter.resource_name,
                "Model" : self.powermeter.modelName,
                "S/N" : self.powermeter.serialNumber,
                "wavelength [nm]" : self.powermeter.wavelength,
                "Power" : self.ui.lblPowermeterMeasPower.text().strip().split(':')[1].strip()
            }, 
            "Pump" : {
                self.ui.leFlowPort.objectName() : self.ui.leFlowPort.text().strip(),
                self.ui.lePressurePort.objectName() : self.ui.lePressurePort.text().strip(),
                self.ui.rbPumpOff.objectName() : self.ui.rbPumpOff.isChecked(),
                self.ui.rbPumpAdjust.objectName() : self.ui.rbPumpAdjust.isChecked(),
                self.ui.rbPumpHold.objectName() : self.ui.rbPumpHold.isChecked(),
                self.ui.rbPumpFlow.objectName() : self.ui.rbPumpFlow.isChecked(),
                self.ui.rbPumpAutomatic.objectName() : self.ui.rbPumpAutomatic.isChecked(),
            },
            "Andor" : {
                self.ui.sbAndorExposure_ms.objectName() : self.ui.sbAndorExposure_ms.value(),
                self.ui.sbAndorEmGain.objectName() : self.ui.sbAndorEmGain.value(),
                self.ui.sbAndorTemp_degC.objectName() : self.ui.sbAndorTemp_degC.value(),
                self.ui.rb32.objectName() : self.ui.rb32.isChecked(),
                self.ui.rb64.objectName() : self.ui.rb64.isChecked(),
                self.ui.rb512.objectName() : self.ui.rb512.isChecked(),
                self.ui.rb16x512fk.objectName() : self.ui.rb16x512fk.isChecked(),
                self.ui.cbAndorBinning.objectName() : self.ui.cbAndorBinning.currentText(),
                "andorCycleTime [s]" : self.andorCycleTime,
                "andorActualExposureTime [s]" : self.actualAndorExpTime,
            },
            "Andor static": {}, # This will hold the Andor setting
            "Stage": {}, # This will hold the Stage setting
        }

        # Gather Stage Table Data (Key,Addr,Axis)
        for row in range(self.ui.twMapping.rowCount()):
            key = self.ui.twMapping.verticalHeaderItem(row).text().strip()
            addrItem = self.ui.twMapping.item(row, self.colAddr)
            axisItem = self.ui.twMapping.item(row, self.colAxis)
            
            addr = self.decodeItem(addrItem)
            axis = self.decodeItem(axisItem)
                
            configData["Stage"][key] = (addr, axis)

        # Gather Andor Table Data (Key,Value)
        for row in range(self.ui.twAndorSetting.rowCount()):
            keyItem = self.ui.twAndorSetting.item(row, 0)
            valItem = self.ui.twAndorSetting.item(row, 1)
            
            if keyItem and keyItem.text().strip():
                key = keyItem.text().strip()
                value = self.decodeItem(valItem)
                    
                configData["Andor static"][key] = value

        # Write to File
        with open(tomlFileName, "wb") as f: # Note the 'wb' for binary write
            tomli_w.dump(configData, f)
            
          
    def saveSetup(self):
        fileDialog = QFileDialog(self)
        fileDialog.setWindowTitle("Save File")
        fileDialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        fileDialog.setViewMode(QFileDialog.ViewMode.Detail)
        fileDialog.setNameFilters(["Setup (*.toml)","All file (*)"])

        if fileDialog.exec():
            self.saveMetadata(fileDialog.selectedFiles()[0])
            

    def startFastKineticMode(self, noFrames, rowHeight, exposureTime, filename):
        self.setupAndorCameraByUserSetting() # for Temp, EM Gain

        # create a queue to pass data
        self.data_queue = queue.Queue()

        # check if we have leftover thread
        if self.acq_worker is not None:
            self.requestRunAcqWorkerFK.disconnect()
            self.acq_worker.abort()

        # Setup Writer Thread
        self.writer_thread = QThread()
        self.writer = TiffWriterWorker(self.data_queue, filename, self.ui.sbAndorExposure_ms.value() * 1e-3, rowHeight)
        self.writer.moveToThread(self.writer_thread)
        self.writer_thread.started.connect(self.writer.process_queue)
        
        # Setup Acquisition Thread
        self.acq_thread = QThread()
        self.acq_worker = AndorFastKineticAcquisitionWorker(self.iXon, self.data_queue)
        self.acq_worker.moveToThread(self.acq_thread)
        # Connect the trigger and abort signals to the worker slot
        self.requestRunAcqWorkerFK.connect(self.acq_worker.run_acquisition)
        # Connections
        self.acq_worker.finished.connect(self.cleanupFK)
        self.acq_worker.progress.connect(self.updateStatusLabel) # Safe cross-thread UI update
        self.acq_worker.error.connect(self.errorFK)
        
        # Start both
        self.writer_thread.start()
        self.acq_thread.start()

        # start the acquisition
        self.requestRunAcqWorkerFK.emit(noFrames, rowHeight, exposureTime)

        self.actualAndorExpTime = exposureTime
        self.andorCycleTime = exposureTime + FAST_KINETICS_FRAME_OVERHEAD
        self.ui.lblAndorActualExposure.setText(f'{exposureTime * 1000}ms / {self.andorCycleTime * 1000:.3f}ms')

        self.fastKineticRunning = True


    @pyqtSlot()
    def cleanupFK(self):
        if self.receivers(self.requestRunAcqWorkerFK) > 0:
            self.requestRunAcqWorkerFK.disconnect()
        else:
            self.log.warning('requestRunAcqWorkerFK has no connection', stack_info = True)
        self.acq_thread.quit()
        self.acq_thread.wait()
        self.writer_thread.quit()
        self.writer_thread.wait()
        self.ui.lblAndorStatus.setText('Done FK')
        self.fastKineticRunning = False
        self.singleFrameAcq = False
        self.acq_worker = None
        self.writer = None


    @pyqtSlot(str)
    def errorFK(self, message):
        self.log.warning('Got error signal from FK thread')
        self.acq_worker.abort()
        self.cleanupFK()


    @pyqtSlot(int, float, float, float, int, np.ndarray)
    def updateStatusLabel(self, burstNo, dataProcessTime, eventCallTime, acqCallTime, bufferIdx, frame):
        self.ui.lblAndorStatus.setText(f'Burst {burstNo}')
        
        self.log.debug(f'Got burst {burstNo} from thread, dataProcessTime {dataProcessTime} eventCallTime {eventCallTime} acqCallTime {acqCallTime} bufferIdx {bufferIdx}')

        self.frameFK = frame


    def onStartExpClicked(self):
        self.expRunning = True
        self.ui.lblFlashOn.setVisible(True)
        self.ui.cbAndorLiveView.setChecked(False)
        elecRunning = False
        # Build the string for folder based on current timestamp and user selected root folder
        timeStamp = datetime.now().strftime("%y%m%d%H%M%S")
        self.saveFolder = self.ui.leDataPath.text() + '\\' + timeStamp + '_' + self.ui.leComment.text().replace(" ", "_")
        self.saveDrive, _ =  os.path.splitdrive(self.saveFolder)
        # try to create the destantion folder or change into it
        try:
            os.makedirs(self.saveFolder, exist_ok = True) # creates the folder and all intermediate-level folders needed to contain it.
        except Exception:
            QMessageBox.critical(self, 'OS Error','Failed to create folder')
            self.expRunning = False
            self.ui.lblFlashOn.setVisible(False)
            return

        if self.iXon is not None: # if we have a functioning Andor camera    
            # first stop the feed from the Andor camera
            ret = self.iXon.AbortAcquisition()
            self.log.debug("Function AbortAcquisition returned {}".format(ERROR_CODE[ret]))

        # turn flow on
        if self.pnumaticPump is not None and self.ui.rbPumpAutomatic.isChecked():
            self.pnumaticPump.setState(FlowPressureState.PUMP)

        if self.socketConnected and self.ui.cbDBAmpEnable.isChecked(): # if a communication to DBAmp is running and user enable it
            if self.ui.cbUseVoltage.isChecked(): # if the user want the application to handle voltage commands
                # set the voltage
                commOK, self.msgIdx = DBAmpSocket.setDrillModeDBamp(self.socketDBAmp, 
                                                                    self.msgIdx, 
                                                                    drillMode = False, 
                                                                    drvVoltage = str(self.ui.sbVoltage_mv.value()), 
                                                                    drvLimit = str(self.ui.sbVoltage_mv.maximum()))
        
            # the file name is the timestamp and the user comment
            elecRecordingFilename = self.saveFolder + '\\' + timeStamp + '.dat'

            if self.ui.leIP.text() != '127.0.0.1': # if the DBAMmp is not on the same computer
                # will use a network folder that has the same hierarchy as the local folder 
                # assume the remote computer has the network mapped to G
                self.networkSaveFolder = self.saveFolder.replace(self.saveDrive, self.networkDrive)
                # try to create the destantion folder on the network or change into it
                try:
                    os.makedirs(self.networkSaveFolder, exist_ok = True) # creates the folder and all intermediate-level folders needed to contain it.
                    # now the DBAmp needs to write to that network folder
                    elecRecordingFilename = elecRecordingFilename.replace(self.saveDrive, self.networkDrive)
                except Exception:
                    QMessageBox.critical(self, 'OS Error','Failed to create network folder')
                    self.expRunning = False
                    self.ui.lblFlashOn.setVisible(False)
                    return

            # send the save command
            commOK, self.msgIdx = DBAmpSocket.setRecDBamp(self.socketDBAmp, 
                                                          self.msgIdx,
                                                          str(self.ui.sbRunTime_s.value()),
                                                          recFileName = elecRecordingFilename,
                                                          usingTrig = self.ui.cbWaitForTrig.isChecked())
            
            if commOK: # if communication was OK
                elecRunning = True
            else:
                self.log.warning('DBAmp Communication failed')
                # notify the user and allow him to continue using the OK button
                choice = QMessageBox.warning(self, 'DBAmp', 'Communication failed', buttons = QMessageBox.StandardButton.Ignore|QMessageBox.StandardButton.Abort)

                if choice == QMessageBox.StandardButton.Abort: # if the user selected to abort this attempt
                    self.expRunning = False
                    self.ui.lblFlashOn.setVisible(False)
                    return
            
        # choose the analog output channel based on user selection for LED source
        if self.ui.rbReflectionLED.isChecked():
            aoCh = 'ao0'
        else:
            aoCh = 'ao1'
        # start the LED flash
        self.startLedFlash(pulseDuration = self.ui.sbPulseDuration.value(), 
                           pulsePeriod = self.ui.sbPulsePeriod.value(),
                           pulseAmp = self.ui.sbPulseVoltage.value(),
                           aoCh = aoCh,
                           useInternalTiming = self.ui.rbInternalTiming.isChecked())
        if not elecRunning: # if the electrical is not running we will never get to the end of exp
            # let the LED stop based on the exp time
            QTimer.singleShot(self.ui.sbRunTime_s.value() * 1000 + 1000, self.onSingleShotEnd)
            self.log.debug('started timer to end LED pulses b/c the DBAmp is not running')

        if self.ui.cbBlinkLaserL2.isChecked():
            try:
                if self.ui.rbAlternatingBlink.isChecked():
                    # set the DAQ to divide the frame clock so it will blink the laser 2 on half of the frames
                    self.laserBlink = ClockDivder(
                        dev_name = 'Dev1',
                        clock_source_pin = 'PFI0',
                        output_line = self.ui.leOutputPin.text(),
                        output_line_inverted = self.ui.leInvertedOutputPin.text(),
                        divisor = self.ui.sbClockDivisor.value()
                    )
                else:
                    self.laserBlink = TriggeredPulseGenerator(
                        dev_name = 'Dev1',
                        trigger_pin = 'PFI0',
                        output_pin = self.ui.leOutputPin.text(),
                        pulse_duration_us = float(self.ui.sbLaserPulseDuration_us.value())
                    )
                    
                self.laserBlink.start()
                
                # set the laser to accept analog modulation 
                self.combinerBox.set_analog_modulation(2, True)
                    
            except ValueError as e:
                self.log.error(f"Configuration Error: {e}")
            except DaqError as e:
                self.log.error("** DAQmx ERROR **")
                self.log.error(f"Details: {e}")
            
        if self.iXon is not None: # if we have a functioning Andor camera
            (ret , status) = self.iXon.GetStatus()
            self.log.debug("Function GetStatus returned {} Status = {}".format(ERROR_CODE[ret], ERROR_CODE[status]))
            
            if status == Status.Idle:
                # get the coordinates of the ROI based on user selection
                self.getFrameBoundaries()

                # the file name is the timestamp
                optRecordingFilename = self.saveFolder + '\\' + timeStamp
                self.optRecordingFilename = optRecordingFilename + '.tif'

                if self.ui.rb16x512fk.isChecked(): # if in Fast Kinetic Mode
                    self.startFastKineticMode(int(self.ui.sbRunTime_s.value() / (self.ui.sbAndorExposure_ms.value() * 1e-3)), 
                                                FK_ROWS,
                                                self.ui.sbAndorExposure_ms.value() * 1e-3,
                                                self.optRecordingFilename)
                else:
                    ret = self.iXon.SetAcquisitionMode(self.iXonCodes.Acquisition_Mode.KINETICS)
                    self.log.debug("Function SetAcquisitionMode returned {} mode = Kinetics".format(ERROR_CODE[ret]))

                    ret = self.iXon.SetIsolatedCropModeEx(Crop.ON,self.frameHeight, self.frameWidth,self.andorBin, self.andorBin, self.andorFrameLeft, self.andorFrameBottom)
                    self.log.debug("Function SetIsolatedCropModeEx returned {} mode = ON".format(ERROR_CODE[ret]))

                    ret = self.iXon.SetIsolatedCropModeType(CropMode.HighSpeed)
                    self.log.debug("Function SetIsolatedCropModeType returned {} mode = high Speed".format(ERROR_CODE[ret]))

                    ret = self.iXon.SetKineticCycleTime(self.ui.sbAndorExposure_ms.value() * 1e-3)
                    self.log.debug("Function SetKineticCycleTime returned {}".format(ERROR_CODE[ret]))

                    self.setupAndorCameraByUserSetting()

                    ret = self.iXon.SetNumberKinetics(int(self.ui.sbRunTime_s.value()/self.andorCycleTime))
                    self.log.debug("Function SetNumberKinetics returned {}".format(ERROR_CODE[ret]))

                    ret = self.iXon.SetSpool(Spool.Enable, self.iXonCodes.Spool_Mode.SPOOL_TO_16_BIT_TIFF, optRecordingFilename, 10)
                    self.log.debug("Function SetSpool returned {} Output file = {}".format(ERROR_CODE[ret], optRecordingFilename))

                    ret = self.iXon.StartAcquisition()
                    self.log.debug("Function StartAcquisition returned {} ".format(ERROR_CODE[ret]))

                if self.andoraView and self.ui.cbAndorPixelTrace.isChecked(): # if the Andor view window exists and the user want pixel trace
                    if self.tiffWatcher and self.tiffWatcher.isRunning():
                        self.log.debug("Stopping existing tiff worker before restart.")
                        self.tiffWatcher.stop()
                        self.tiffWatcher.wait()
                    
                    self.tiffWatcher = TiffSpoolWatcher(self.optRecordingFilename, self.andoraView.crossLocation)
                    self.tiffWatcher.dataBlockReady.connect(self.onNewTraceBlock)
                    self.tiffWatcher.errorOccurred.connect(self.onTiffWatcherError)
                    self.tiffWatcher.start()

        # save metadata to the folder
        self.saveMetadata(self.saveFolder + '\\' + timeStamp + '_metadata.txt')

    
    def robustNetworkMove(self, src, dst):
        try:
            # Check for the folder  
            if not os.path.exists(src):
                self.log.error(f"Network path {src} unreachable before start.")
                return
                
            # Count files on the network
            filesToMove = os.listdir(src)
            if not filesToMove:
                self.log.warning(f"Source folder {src} is empty. Nothing to move.")
                return

            # Copy the files
            shutil.copytree(src, dst, dirs_exist_ok = True)

            # Verify the files exist locally
            # Check if EVERY file from the network now exists locally
            successfulMove = all(os.path.exists(os.path.join(dst, f)) for f in filesToMove)

            if successfulMove:
                # DELETE only after verification
                shutil.rmtree(src)
            else:
                self.log.error(f"Verification FAILED: Some files did not arrive from folder {src}.")

        except Exception as e:
            self.log.error(f"Unexpected failure: {e}")


    def onAbortClicked(self):
        if self.acq_worker is not None:
            self.acq_worker.abort()

        if self.expRunning: # we are waiting for the experiment to end
            # update status
            self.expRunning = False
            self.ui.lblFlashOn.setVisible(False)

            # turn LED OFF
            self.stopLED()

            # turn flow off
            if self.pnumaticPump is not None and self.ui.rbPumpAutomatic.isChecked():
                self.pnumaticPump.setState(FlowPressureState.HOLD)
            
            # stop laser blinking if it was used
            if self.laserBlink is not None:
                self.laserBlink.close()
                self.laserBlink = None

            # disable analog modulation on laser 2
            if self.combinerBox is not None:
                self.combinerBox.set_analog_modulation(2, False)

            if self.socketConnected:
                commOK, self.msgIdx, recStatus = DBAmpSocket.getRecStatDBamp(self.socketDBAmp, self.msgIdx)
                if recStatus: # if recording 
                    commOK, self.msgIdx = DBAmpSocket.stopRecDBamp(self.socketDBAmp, self.msgIdx)

                if self.ui.cbUseVoltage.isChecked(): # if the user want the application to handle voltage commands
                    # turn votlage to zero
                    commOK, self.msgIdx = DBAmpSocket.setDrillModeDBamp(self.socketDBAmp, self.msgIdx, drillMode = False, drvVoltage = '0')

                if self.ui.leIP.text() != '127.0.0.1': # if the DBAmp is not on the same computer
                    self.robustNetworkMove(self.networkSaveFolder, self.saveFolder)             

            if self.iXon:
                ret = self.iXon.AbortAcquisition()
                self.log.debug("Function StartAcquisition returned {} ".format(ERROR_CODE[ret]))

                if self.tiffWatcher and self.tiffWatcher.isRunning():
                    self.tiffWatcher.stop()
                    self.tiffWatcher.wait()

            self.log.info('Experiment Aborted')


    def startLedFlash(self,
                        pulseDuration = 10.0,
                        pulsePeriod = 10000.0,
                        pulseAmp = 5.0,
                        aoCh = 'ao0',
                        useInternalTiming = True
    ):
        if pulsePeriod <= pulseDuration:
            self.log.warning(f"LED's pulse duration ({pulseDuration}) is smaller than period ({pulsePeriod})")
            pulseDuration = pulsePeriod + 1

        if useInternalTiming:
            self.pulseGen.setupInternalClockPulseGeneration(aoCh,
                                                            pulseAmp,
                                                            pulseDuration,
                                                            pulsePeriod)
        else:
            self.pulseGen.setupExternalClockPulseGeneration(aoCh,
                                                            'PFI0',
                                                            pulseAmp,
                                                            pulseDuration,
                                                            pulsePeriod)
            
        # Start generation
        self.pulseGen.startGeneration()


    def updateTimingUnitsLED(self):
        durationFactor = 1
        periodFactor = 1
        if self.ui.rb16x512fk.isChecked():
            if self.ui.sbPulseDuration.suffix() == ' frames':
                durationFactor = 0.1
                periodFactor = 0.01
            suffix = " bursts"
        else:
            if self.ui.sbPulseDuration.suffix() == ' bursts':
                durationFactor = 10
                periodFactor = 100
            suffix = " frames"

        if self.ui.rbSyncFrameTiming.isChecked():
            self.ui.sbPulseDuration.setSuffix(suffix)
            self.ui.sbPulsePeriod.setSuffix(suffix)

            self.ui.sbPulseDuration.setValue(int(self.ui.sbPulseDuration.value() * durationFactor))
            self.ui.sbPulsePeriod.setValue(int(self.ui.sbPulsePeriod.value() * periodFactor))
        else:
            self.ui.sbPulseDuration.setSuffix(' ms')
            self.ui.sbPulsePeriod.setSuffix(' ms')


    def stopLED(self):
        self.pulseGen.stopGeneration()

    def onSelecDataPathClicked(self):
        # initial setting for the options
        options = QFileDialog.Option(0)
        # Shows only directories
        options |= QFileDialog.Option.ShowDirsOnly  
        dataFolderPath = QFileDialog.getExistingDirectory(self, "Select folder for data", self.ui.leDataPath.text(), options=options)

        if dataFolderPath:
            self.ui.leDataPath.setText(dataFolderPath)


    def onPulseDurationChange(self):
        self.ui.sbPulsePeriod.setMinimum(self.ui.sbPulseDuration.value() + 1)


    def onSliderAndorEmGainChange(self):
        self.ui.sbAndorEmGain.setValue(self.ui.sldAndorEmGain.value())
        self.ui.lblSettingPending.setVisible(True)


    def onAndorEmGainChange(self):
        self.ui.sldAndorEmGain.setValue(self.ui.sbAndorEmGain.value())
        self.ui.lblSettingPending.setVisible(True)


    def OnAndorExpTimeChange(self):
        self.ui.lblSettingPending.setVisible(True)


    def onAndorTempChange(self):
        self.ui.lblSettingPending.setVisible(True)


    def onAndorRoiToggled(self):
        self.ui.lblSettingPending.setVisible(True)

        self.updateTimingUnitsLED()


    def onAndorBinChnage(self):
        self.ui.lblSettingPending.setVisible(True)


    def onInternalTimingToggle(self):
        self.ui.lblSettingPending.setVisible(True)
        
        self.updateTimingUnitsLED()


    def onPowerSpinBoxFisnihedL1(self):
        """
        Handles the value changed event for the power spin box of laser 1.

        This method is triggered when the user adjusts the spin box for laser 1's power.
        It updates the corresponding power slider to match the spin box's value and,
        if a combiner box is connected, sends the new power setpoint to the combiner
        for laser line 1.

        Args:
            None 

        Returns:
            None
        """
        self.ui.sldPowerL1.setValue(int(self.ui.sbPowerL1_mw.value() * self.powerPrecisionFactorL1))

        if self.combinerBox is not None:
            self.combinerBox.set_power_setpoint(1, self.ui.sbPowerL1_mw.value())


    def onPowerSpinBoxFisnihedL2(self):
        """
        Handles the value changed event for the power spin box of laser 2.

        This method is triggered when the user adjusts the spin box for laser 2's power.
        It updates the corresponding power slider to match the spin box's value and,
        if a combiner box is connected, sends the new power setpoint to the combiner
        for laser 2.

        Args:
            None 

        Returns:
            None
        """
        self.ui.sldPowerL2.setValue(int(self.ui.sbPowerL2_mw.value() * self.powerPrecisionFactorL2))

        if self.combinerBox is not None:
            self.combinerBox.set_power_setpoint(2, self.ui.sbPowerL2_mw.value())


    def onPowerSpinBoxFisnihedL3(self):
        """
        Handles the value changed event for the power spin box of laser 3.

        This method is triggered when the user adjusts the spin box for laser 3's power.
        It updates the corresponding power slider to match the spin box's value and,
        if a combiner box is connected, sends the new power setpoint to the combiner
        for laser 3.

        Args:
            None 

        Returns:
            None
        """
        self.ui.sldPowerL3.setValue(int(self.ui.sbPowerL3_mw.value() * self.powerPrecisionFactorL3))

        if self.combinerBox is not None:
            self.combinerBox.set_power_setpoint(3, self.ui.sbPowerL3_mw.value())


    def onPowerSliderChangedL1(self):
        """
        Handles the value changed event for the power slider of laser 1.

        This method is triggered when the user adjusts the slider for laser 1's power.
        It updates the corresponding power spin box to match the slider's value,
        ensuring that both UI elements reflect the same current power setting.

        Args:
            None 

        Returns:
            None
        """
        self.ui.sbPowerL1_mw.setValue(self.ui.sldPowerL1.value() / self.powerPrecisionFactorL1)


    def onPowerSliderChangedL2(self):
        """
        Handles the value changed event for the power slider of laser 2.

        This method is triggered when the user adjusts the slider for laser 2's power.
        It updates the corresponding power spin box to match the slider's value,
        ensuring that both UI elements reflect the same current power setting.

        Args:
            None 

        Returns:
            None
        """
        self.ui.sbPowerL2_mw.setValue(self.ui.sldPowerL2.value() / self.powerPrecisionFactorL2)


    def onPowerSliderChangedL3(self):
        """
        Handles the value changed event for the power slider of laser 3.

        This method is triggered when the user adjusts the slider for laser 3's power.
        It updates the corresponding power spin box to match the slider's value,
        ensuring that both UI elements reflect the same current power setting.

        Args:
            None 

        Returns:
            None
        """
        self.ui.sbPowerL3_mw.setValue(self.ui.sldPowerL3.value() / self.powerPrecisionFactorL3)


    def onLaserPowerSpinBoxFisnihedL1(self):
        """
        Handles the value changed event for the laser power spin box of laser 1.

        This method is triggered when the user adjusts the spin box for laser 1's power.
        if a combiner box is connected, sends the new power setpoint to the combiner
        for laser 1.

        Args:
            None 

        Returns:
            None
        """

        if self.combinerBox is not None:
            self.combinerBox.set_laser_power_setpoint(1, self.ui.sbLaserPowerL1_mw.value())


    def onLaserPowerSpinBoxFisnihedL2(self):
        """
        Handles the value changed event for the laser power spin box of laser 2.

        This method is triggered when the user adjusts the spin box for laser 2's power.
        if a combiner box is connected, sends the new power setpoint to the combiner
        for laser 2.

        Args:
            None 

        Returns:
            None
        """

        if self.combinerBox is not None:
            self.combinerBox.set_laser_power_setpoint(2, self.ui.sbLaserPowerL2_mw.value())


    def onLaserPowerSpinBoxFisnihedL3(self):
        """
        Handles the value changed event for the laser power spin box of laser 3.

        This method is triggered when the user adjusts the spin box for laser 3's power.
        if a combiner box is connected, sends the new power setpoint to the combiner
        for laser 3.

        Args:
            None 

        Returns:
            None
        """

        if self.combinerBox is not None:
            self.combinerBox.set_laser_power_setpoint(3, self.ui.sbLaserPowerL3_mw.value())

    
    def setLaserEmission(self, laserNo, stateEnable):
        """
        laser emission control

        This method is setting the laser emission state

        Args:
            laserNo (int) : The number of the laser to update
            stateEnable (boolean) : True for emssion ON 

        Returns:
            None
        """

        if stateEnable:
            self.combinerBox.enable(laserNo)
        else:
            self.combinerBox.disable(laserNo)


    def onEnableToggleL1(self):
        """
        Handles the toggled state of the enable checkbox for laser 1.

        This method is invoked when the user clicks the enable checkbox for laser 1.
        If a combiner box is connected, it either enables or disables laser 1
        based on the checkbox's current checked state.

        Args:
            None 

        Returns:
            None
        """
        if self.combinerBox is not None:
            self.setLaserEmission(1, self.ui.cbEnableL1.isChecked())

    
    def onEnableToggleL2(self):
        """
        Handles the toggled state of the enable checkbox for laser 2.

        This method is invoked when the user clicks the enable checkbox for laser 2.
        If a combiner box is connected, it either enables or disables laser 2
        based on the checkbox's current checked state.

        Args:
            None 

        Returns:
            None
        """
        if self.combinerBox is not None:
            self.setLaserEmission(2, self.ui.cbEnableL2.isChecked())

        
    def onEnableToggleL3(self):
        """
        Handles the toggled state of the enable checkbox for laser 3.

        This method is invoked when the user clicks the enable checkbox for laser 3.
        If a combiner box is connected, it either enables or disables laser 3
        based on the checkbox's current checked state.

        Args:
            None 

        Returns:
            None
        """
        if self.combinerBox is not None:
            self.setLaserEmission(3, self.ui.cbEnableL3.isChecked())

        
    def onStartSearchClicked(self):
        self.searchCycle = 1
        self.searchLegNumber = 1


    def onStopSearchClicked(self):
        self.searchLegNumber = 0


    def onLaserBlinkModeToggled(self, isChecked):
        self.ui.sbClockDivisor.setEnabled(isChecked)
        self.ui.leInvertedOutputPin.setEnabled(isChecked)
        self.ui.sbLaserPulseDuration_us.setEnabled(not isChecked)


    def onUseVoltageToggled(self, isChecked):
        self.ui.sbVoltage_mv.setEnabled(isChecked)


    def onPumpModeToggled(self):
        if self.pnumaticPump is not None:
            if self.ui.rbPumpOff.isChecked():
                self.pnumaticPump.setState(FlowPressureState.OFF)
            elif self.ui.rbPumpAdjust.isChecked():
                self.pnumaticPump.setState(FlowPressureState.ADJUST)
            elif self.ui.rbPumpHold.isChecked():
                self.pnumaticPump.setState(FlowPressureState.HOLD)
            elif self.ui.rbPumpFlow.isChecked():
                self.pnumaticPump.setState(FlowPressureState.PUMP)
            elif self.ui.rbPumpAutomatic.isChecked(): 
                if not self.expRunning:
                    self.pnumaticPump.setState(FlowPressureState.HOLD)
            else:
                self.log.error('Unknown pump mode')


    def onPumpLineChanged(self):
        if self.pnumaticPump is not None:
            self.pnumaticPump.close()
            self.pnumaticPump = None

        self.pnumaticPump = FlowPressureController(flowLine = self.ui.leFlowPort.text(), pressureLine = self.ui.lePressurePort.text())


class CheckboxItem(QGraphicsProxyWidget):
    def __init__(self, checkboxLabel, parent=None):
        super().__init__(parent)
        self.checkbox = QCheckBox(checkboxLabel)
        self.setWidget(self.checkbox)


class WideFieldView(pg.GraphicsLayoutWidget):
    def __init__(self,parent):
        super().__init__()

        self.crossLocation = [0, 0]
        self.indexX = np.array([])
        self.indexY = np.array([])

        # set background
        self.setBackground("w")
        # set the title
        self.setWindowTitle('Wide Field Camera')

        self.createImageDisplay()


    def createImageDisplay(self):
        # add a plot widget for X crosssection
        self.plotWidgetCrosssectionX = self.addPlot(col = 1, row = 0)
        # add grid
        self.plotWidgetCrosssectionX.showGrid(x = True, y = True)
        # add axis  (left, top, right, bottom)
        self.plotWidgetCrosssectionX.showAxes((True, True, True, False),(False, True, False, False))
        # create empty plot
        self.plotCrosssectionX = self.plotWidgetCrosssectionX.plot([], [], pen = mkPen('b', width = 3))

        # add a plot widget for Y crosssection
        self.plotWidgetCrosssectionY = self.addPlot(col = 0,row = 1)
        # flip Y to match image view
        self.plotWidgetCrosssectionY.invertY()
        # flip X to have the zero next to the image
        self.plotWidgetCrosssectionY.invertX()
        # add grid
        self.plotWidgetCrosssectionY.showGrid(x = True, y = True)
        # add axis  (left, top, right, bottom)
        self.plotWidgetCrosssectionY.showAxes((True, True, False, True),(True, False, False, False))
        # create empty plot
        self.plotCrosssectionY = self.plotWidgetCrosssectionY.plot([], [], pen = mkPen('b', width = 3))

        # add image item in plot widget
        self.plotWidgetImage = self.addPlot(row = 1, col = 1)
        self.plotWidgetImage.hideAxis('bottom')
        self.plotWidgetImage.hideAxis('left')
        self.imageViewer = pg.ImageItem()
        self.plotWidgetImage.addItem(self.imageViewer)
        # configure view for images
        self.plotWidgetImage.getViewBox().setAspectLocked()
        self.plotWidgetImage.invertY()
        # add cross
        self.crossV = pg.InfiniteLine()
        self.plotWidgetImage.addItem(self.crossV)
        self.crossH = pg.InfiniteLine(angle = 0)
        self.plotWidgetImage.addItem(self.crossH)
        # connect a mouse click event for the cross location selection
        self.plotWidgetImage.scene().sigMouseClicked.connect(self.onImageMouseClicked)
        # Connect to the view range changed signal
        self.plotWidgetImage.getViewBox().sigRangeChanged.connect(self.onImageRangeChanged)

        # create a mono histogram item
        cbar = pg.HistogramLUTItem(image = self.imageViewer)
        cbar.gradient.loadPreset('grey')
        cbar.setHistogramRange(0, 255)
        cbar.setLevels(0, 255)
        # Hide the ticks
        for tick in cbar.gradient.ticks:
            tick.setVisible(False)
        # add it to the window
        self.addItem(cbar, col = 2,row = 1)

        # create a checkbox for AutoLevel
        self.cbItemAutoLevel = CheckboxItem('Auto level')
        self.cbItemAutoLevel.checkbox.setStyleSheet("QCheckBox { background-color: transparent; }")
        self.addItem(self.cbItemAutoLevel, col = 2, row = 2)
        self.cbAutoLevel = self.cbItemAutoLevel.checkbox

        # create a label for the extra info
        self.lblInfo = self.addLabel('<font color="black">Span X: xxxxx<br>Span Y: xxxxx</font>', col = 0, row = 0)
        # create a lebel for the cross information
        self.lblCrossInfo = self.addLabel(' X, Y : xxxxx', col = 1,row = 2)

        # set the proporation so the image take most of the display
        qGraphicsGridLayout = self.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(0, 1)
        qGraphicsGridLayout.setColumnStretchFactor(1, 5)
        qGraphicsGridLayout.setRowStretchFactor(0, 1)
        qGraphicsGridLayout.setRowStretchFactor(1, 5)


    def updateCross(self, crossLocation):
        # remember the cross location
        self.crossLocation = crossLocation
        # update the cross markers to the middle of the pixel
        self.crossV.setPos(self.crossLocation[0] + 0.5)
        self.crossH.setPos(self.crossLocation[1] + 0.5)
        # update the label
        self.lblCrossInfo.setText(str(crossLocation[0]) + ',' + str(crossLocation[1]) + ':')


    @pyqtSlot(object)
    def onImageMouseClicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # Left mouse button
            # gte the event positions
            pos = event.scenePos()
            # Map scene to image coordinates
            imagePos = self.imageViewer.mapFromScene(pos) 
            # make sure the cross is within the frame
            clampedX = int(max(min(imagePos.x(), self.imageViewer.width() - 1), 0))
            clampedY = int(max(min(imagePos.y(), self.imageViewer.height() - 1), 0))

            # update the display of the cross
            self.updateCross([clampedX, clampedY])


    @pyqtSlot(object, object)
    def onImageRangeChanged(self, viewbox, range):
        # get the new range
        xMin, xMax, yMin, yMax = range[0][0], range[0][1], range[1][0], range[1][1]

        # update the corsssection ranges to match 
        self.plotWidgetCrosssectionX.setXRange(xMin, xMax, padding = 0)
        self.plotWidgetCrosssectionY.setYRange(yMin, yMax, padding = 0)


    def setNewImage(self,frame):
        # update the display
        self.imageViewer.setImage(frame, autoLevels = self.cbAutoLevel.isChecked())

        # get current range
        xMin , xMax = self.plotWidgetImage.viewRange()[0]
        yMin , yMax = self.plotWidgetImage.viewRange()[1]

        # calculate the matching index in the frame
        xMinIdx = int(max(xMin, 0))
        xMaxIdx = int(max(min(xMax,frame.shape[0] - 1), xMinIdx + 1))
        yMinIdx = int(max(yMin, 0))
        yMaxIdx = int(max(min(yMax,frame.shape[1] - 1), yMinIdx + 1))

        # check if the cross is outside the new frame 
        if self.crossLocation[0] > frame.shape[0] - 1 : # if the cross is outside the new frame
            self.crossLocation[0] = frame.shape[0] - 1
            self.updateCross([frame.shape[0] - 1, self.crossLocation[1]])
        if self.crossLocation[1] > frame.shape[1] - 1 : # if the cross is outside the new frame
            self.crossLocation[1] = frame.shape[1] - 1
            self.updateCross([self.crossLocation[0], frame.shape[1] - 1])

        # check if the new frame has different size
        if frame.shape[0] != self.indexX.shape[0] : # if the new frame has different width
            self.indexX = np.arange(frame.shape[0])
        if frame.shape[1] != self.indexY.shape[0] : # if the new frame has different height
            self.indexY = np.arange(frame.shape[1])

        # update the crosssection plots
        self.plotCrosssectionX.setData(self.indexX, frame[:, self.crossLocation[1]])
        #self.plotWidgetCrosssectionX.setXRange(self.indexX[0],self.indexX[-1])
        self.plotCrosssectionY.setData(frame[self.crossLocation[0], :], self.indexY)
        #self.plotWidgetCrosssectionY.setYRange(self.indexY[0],self.indexY[-1])
        # display the Max-Min as an axis title
        spanX = int(np.max(frame[xMinIdx:xMaxIdx, self.crossLocation[1]]) - np.min(frame[xMinIdx:xMaxIdx, self.crossLocation[1]]))
        spanStrX = '| ' + str(spanX) + ' |'
        self.plotWidgetCrosssectionX.setLabel('left',spanStrX)
        spanY = int(np.max(frame[self.crossLocation[0], yMinIdx:yMaxIdx]) - np.min(frame[self.crossLocation[0], yMinIdx:yMaxIdx]))
        spanStrY = '| ' + str(spanY) + ' |'
        self.plotWidgetCrosssectionY.setLabel('top',spanStrY)

        # update the span label and keep it in a constant length
        self.lblInfo.setText(f'<font color="black">Span X:{spanX:5}<br>Span Y:{spanY:5}</font>')
        
        # update the cross label
        self.lblCrossInfo.setText(str(self.crossLocation[0]) + ' , ' + str(self.crossLocation[1]) + ' : ' + str(int(frame[self.crossLocation[0], self.crossLocation[1]])))


    def viewFullImage(self):
        self.plotWidgetImage.getViewBox().setRange(xRange = (0, self.imageViewer.width()), yRange = (0, self.imageViewer.height()))


    def addGraphicalROI(self):
        # get the current visibale range
        viewRange = self.plotWidgetImage.viewRange()
        # calculate the position and size for centred ROI
        roiLeft = (viewRange[0][1] - viewRange[0][0]) * 0.4 + viewRange[0][0] # set to 10 % left of center
        roiTop = (viewRange[1][1] - viewRange[1][0]) * 0.4 + viewRange[1][0] # set to 10 % left of center
        roiWidth = (viewRange[0][1] - viewRange[0][0]) * 0.2 # set width to 20% of visibale range
        roiHeight = (viewRange[1][1] - viewRange[1][0]) * 0.2 # set width to 20% of visibale range
        # create the ROI object
        self.roi = pg.RectROI([roiLeft, roiTop], [roiWidth, roiHeight], pen = 'g')
        # set bigger handles
        for handle in self.roi.getHandles():
            handle.radius = 10
            handle.buildPath()
            handle.update()
        # add it to the plot
        self.plotWidgetImage.addItem(self.roi)


    def getGraphicalROI(self):
        if self.roi:
            return self.roi.parentBounds()
        else:
            return None


    def removeGraphicalROI(self):
        if self.roi: # if the ROI object exists
            # remove the item from the display
            self.plotWidgetImage.removeItem(self.roi)
            # delete the object
            self.roi = None


class AndorView(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()

        self.crossLocation = [0, 0]

        # set title
        self.setWindowTitle('Andor view')
        # set background
        self.setBackground("w")

        self.createImageDisplay()


    def createImageDisplay(self):   
        # styles to use later
        labelStyle = {"color": "black", "font-size": "10pt"}
        tickLabelFont = QFont("Calibri", 10)

        # create a plot for the pixel over time
        self.plotWidgetPixel = self.addPlot(row = 0 , col = 0, colspan = 3)
        self.pixelPlot = self.plotWidgetPixel.plot(np.array([]), np.array([]), pen = mkPen('b', width = 3))
        # Set the axis label 
        self.plotWidgetPixel.getAxis("left").setLabel("Counts", **labelStyle)
        self.plotWidgetPixel.getAxis("bottom").setLabel("Frames", **labelStyle)
        # Apply a font to the tick labels
        self.plotWidgetPixel.getAxis("left").setStyle(tickFont = tickLabelFont)
        self.plotWidgetPixel.getAxis("bottom").setStyle(tickFont = tickLabelFont)
        # add grid
        self.plotWidgetPixel.showGrid(x = True, y = True)
        # set x range
        self.plotWidgetPixel.setXRange(0, param_maximumPixelHistory)

        # add a item to hold the image
        self.plotWidgetFrame = self.addPlot(row = 1, col = 0, colspan = 2)
        self.plotWidgetFrame.hideAxis('bottom')
        self.plotWidgetFrame.hideAxis('left')
        # Get the ViewBox for the plot
        self.viewbox = self.plotWidgetFrame.getViewBox()

        # create the image ovject
        self.frameViewer = pg.ImageItem()
        # add it to the plot
        self.plotWidgetFrame.addItem(self.frameViewer)
        # configure view for images
        self.plotWidgetFrame.getViewBox().setAspectLocked()
        self.plotWidgetFrame.invertY()

        # add cross
        self.crossV = pg.InfiniteLine()
        self.plotWidgetFrame.addItem(self.crossV)
        self.crossH = pg.InfiniteLine(angle = 0)
        self.plotWidgetFrame.addItem(self.crossH)
        # connect a mouse click event for the cross location selection
        self.plotWidgetFrame.scene().sigMouseClicked.connect(self.onImageMouseClicked)

        # create a lebel for the cross information
        self.lblCrossInfo = self.addLabel(' X, Y : xxxxx', col = 1, row = 2)

        # Create a PlotDataItem for a square as 64x64 crop target
        self.square64 = pg.PlotDataItem([], [],
                                    pen = pg.mkPen('r', width = 2),
                                    connect = 'all') 
        self.plotWidgetFrame.addItem(self.square64)
        # Create a TextItem for the square size
        self.text64 = pg.TextItem(text = '64x64', color = (255, 255, 0)) # Yellow text
        # locate it in the top left corner of the square
        self.text64.setPos(ROIcoordinates.CropCenter64x64[0], ROIcoordinates.CropCenter64x64[2])
        self.text64.setAnchor((0, 1)) # Anchor text in the left bottom
        self.plotWidgetFrame.addItem(self.text64)
        # hide it for now
        self.square64.setVisible(False)
        self.text64.setVisible(False)

        # Create a PlotDataItem for a square as 32x32 crop target
        self.square32 = pg.PlotDataItem([], [],
                                    pen = pg.mkPen('r', width = 2),
                                    connect = 'all') 
        self.plotWidgetFrame.addItem(self.square32)
        # Create a TextItem for the square size
        self.text32 = pg.TextItem(text = '32x32', color = (255, 255, 0)) # Yellow text
        # locate it in the top left corner of the square
        self.text32.setPos(ROIcoordinates.CropCenter32x32[0], ROIcoordinates.CropCenter32x32[2])
        self.text32.setAnchor((0, 1)) # Anchor text in the left bottom
        self.plotWidgetFrame.addItem(self.text32)
        # hide it for now
        self.square32.setVisible(False)
        self.text32.setVisible(False)

        # Connect to the ViewBox's sigRangeChanged signal so we could scale the text
        self.viewbox.sigRangeChanged.connect(self.onImageZoom)

        # create a checkbox for Crop target visibility
        self.cbCropTargetVisibleItem = CheckboxItem('Show crop targets')
        self.cbCropTargetVisibleItem.checkbox.setStyleSheet("QCheckBox { background-color: transparent; }")
        self.addItem(self.cbCropTargetVisibleItem, col = 0, row = 2)
        self.cbCropTargetVisible = self.cbCropTargetVisibleItem.checkbox

        # create a mono histogram item
        cbar = pg.HistogramLUTItem(image = self.frameViewer)
        cbar.gradient.loadPreset('grey')
        cbar.setHistogramRange(0,64000)
        cbar.setLevels(0,64000)
        # Hide the ticks
        for tick in cbar.gradient.ticks:
            tick.setVisible(False)
        # add it to the window
        self.addItem(cbar, col = 2, row = 1)

        # create a checkbox for AutoLevel
        self.cbItemAutoLevel = CheckboxItem('Auto level')
        self.cbItemAutoLevel.checkbox.setStyleSheet("QCheckBox { background-color: transparent; }")
        self.addItem(self.cbItemAutoLevel, col = 2, row = 2)
        self.cbAutoLevel = self.cbItemAutoLevel.checkbox

        # set the proportions for the histogram vs the optical
        qGraphicsGridLayout = self.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(0, 1)
        qGraphicsGridLayout.setColumnStretchFactor(1, 8)
        qGraphicsGridLayout.setColumnStretchFactor(2, 1)
        qGraphicsGridLayout.setRowStretchFactor(0, 1)
        qGraphicsGridLayout.setRowStretchFactor(1, 3)


    def setCropTargetSize(self, frameWidth):
        # Coordinates for a 64x64 square's perimeter 
        squareCoordsX = np.array([ROIcoordinates.CropCenter64x64[0],
                                    ROIcoordinates.CropCenter64x64[1],
                                    ROIcoordinates.CropCenter64x64[1],
                                    ROIcoordinates.CropCenter64x64[0],
                                    ROIcoordinates.CropCenter64x64[0]])
        squareCoordsY = np.array([ROIcoordinates.CropCenter64x64[2],
                                    ROIcoordinates.CropCenter64x64[2],
                                    ROIcoordinates.CropCenter64x64[3],
                                    ROIcoordinates.CropCenter64x64[3],
                                    ROIcoordinates.CropCenter64x64[2]])
        
        if frameWidth == 256: # if there is 2x binning
            squareCoordsX = squareCoordsX / 2    
            squareCoordsY = squareCoordsY / 2 
        
        # update the square
        self.square64.setData(squareCoordsX, squareCoordsY)
        # update the label
        self.text64.setPos(squareCoordsX[0], squareCoordsY[0])

        # Coordinates for a 32x32 square's perimeter 
        squareCoordsX = np.array([ROIcoordinates.CropCenter32x32[0],
                                    ROIcoordinates.CropCenter32x32[1],
                                    ROIcoordinates.CropCenter32x32[1],
                                    ROIcoordinates.CropCenter32x32[0],
                                    ROIcoordinates.CropCenter32x32[0]])
        squareCoordsY = np.array([ROIcoordinates.CropCenter32x32[2],
                                    ROIcoordinates.CropCenter32x32[2],
                                    ROIcoordinates.CropCenter32x32[3],
                                    ROIcoordinates.CropCenter32x32[3],
                                    ROIcoordinates.CropCenter32x32[2]])
        
        if frameWidth == 256: # if there is 2x binning
            squareCoordsX = squareCoordsX / 2    
            squareCoordsY = squareCoordsY / 2 
        
        # update the square
        self.square32.setData(squareCoordsX, squareCoordsY)
        # update the label
        self.text32.setPos(squareCoordsX[0], squareCoordsY[0])
        

    def updateFrame(self,frame):
        # check if the cross is outside the new frame 
        if self.crossLocation[0] > frame.shape[0] - 1 : # if the cross is outside the new frame
            self.crossLocation[0] = frame.shape[0] - 1
        if self.crossLocation[1] > frame.shape[1] - 1 : # if the cross is outside the new frame
            self.crossLocation[1] = frame.shape[1] - 1
        # update the cross
        self.updateCross(self.crossLocation)

        # set the new image
        # invert horizontally so that it matches wide-field camera
        frame = np.flipud(frame)
        self.frameViewer.setImage(frame, autoLevels = self.cbAutoLevel.isChecked())

        # update the cross label
        crossInfo = f'<font color="black">{self.crossLocation[0]}, {self.crossLocation[1]}: {frame[self.crossLocation[0], self.crossLocation[1]]:.0f}</font>'
        self.lblCrossInfo.setText(crossInfo)

        # update the font 
        self.onImageZoom(self.viewbox, None)

        # update the squares
        self.setCropTargetSize(frame.shape[0])

        # if the frame is larger than the targets for crop - show the target
        self.text64.setVisible(frame.shape[0] > 64 and self.cbCropTargetVisible.isChecked())
        self.square64.setVisible(frame.shape[0] > 64 and self.cbCropTargetVisible.isChecked())
        self.text32.setVisible(frame.shape[0] > 64 and self.cbCropTargetVisible.isChecked())
        self.square32.setVisible(frame.shape[0] > 64 and self.cbCropTargetVisible.isChecked())


    def updateCross(self, crossLocation):
        # remember the cross location
        self.crossLocation = crossLocation
        # update the cross markers to the middle of the pixel
        self.crossV.setPos(self.crossLocation[0] + 0.5)
        self.crossH.setPos(self.crossLocation[1] + 0.5)
        # update the label
        self.lblCrossInfo.setText(str(crossLocation[0]) + ',' + str(crossLocation[1]) + ':')


    @pyqtSlot(object)
    def onImageMouseClicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # Left mouse button
            if self.frameViewer.width() is not None:
                # get the event positions
                pos = event.scenePos()
                # Map scene to image coordinates
                imagePos = self.frameViewer.mapFromScene(pos) 
                # make sure the cross is within the frame
                clampedX = int(max(min(imagePos.x(), self.frameViewer.width() -1 ), 0))
                clampedY = int(max(min(imagePos.y(), self.frameViewer.height() - 1), 0))

                # update the display of the cross
                self.updateCross([clampedX, clampedY])


    @pyqtSlot(object, object)
    def onImageZoom(self, viewboxInstance, range):
        # query the current range directly
        _, yRange = viewboxInstance.viewRange()
        height = yRange[1] - yRange[0]

        # Avoid division by zero if height becomes very small
        if height > 0:
            # calculate new font size and clamp it
            fotnSize = int(max(5, min(50, 1000 / height)))
        else:
            fotnSize = 1

        # create a font with this size
        font = QFont()
        font.setPointSize(fotnSize)

        # assign the font to the text
        self.text64.setFont(font)
        self.text32.setFont(font)


    def updatePixelPlot(self, pixelOverTime):
        self.pixelPlot.setData(np.arange(pixelOverTime.size),pixelOverTime)


def handleCommandLineOptions():
    parser = argparse.ArgumentParser(description='program to run a multi instrument experiment',
                                     usage='''
        This program communicates with the following:
        -8742 Picomotor Controller Driver for the stage
        -Wide field camera
        -Andor iXon camera
        -DBAmp electrical amplifier
        -NI DAQ card

        It shows the feed from both cameras and on command start saving a fixed interval in the Andor and DBAmp while flashing the LED

        To get the full list of possible arguments, run it with -h
                                     ''')
    
    parser.add_argument('--verbose', '-v', type = int, default = 1, help = 'Level of printing debug information to standard output (default to 1 eg warning and above)')
    parser.add_argument('--host', type = str, default = '127.0.0.1',  help = 'DBAmp IP address (default is 127.0.0.1)')
    parser.add_argument('--port', type = int, default = 10285, help = 'DBAmp application IPp ort numbers (default 10285)')
    parser.add_argument('--network_drive', type = str, default = 'G:',  help = 'Network drive to use if DBAmp is not local (default is G:)')

    parser.add_argument('--dummy', action = "store_true", help = 'Create dummy feed for both cameras')    
    
    return parser.parse_args()

# to be able to catch  "Silent" GUI Errors
# In PyQt, if an exception occurs inside a Slot (e.g., a function connected to a button click), the application often swallows the error
ORIGINAL_HOOK = sys.excepthook

def verboseExceptionHook(exctype, value, traceback):
    print(exctype, value, traceback)
    ORIGINAL_HOOK(exctype, value, traceback)

sys.excepthook = verboseExceptionHook

# main
if __name__ == '__main__':
    # get command line options
    args = handleCommandLineOptions()
    # create a QT application
    app = QApplication(sys.argv)
    # create the main window with the command line options
    mainWin = AppWindow(args)
    # show the window
    mainWin.show()
    # run the QT event loop
    sys.exit(app.exec())