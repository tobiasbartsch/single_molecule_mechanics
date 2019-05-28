import pandas as pd
import numpy as np
import xarray as xr
import nptdms

class SingleMoleculeAnalyzer:
    """Provides data structures and methods to analyze single-molecule data."""

    def __init__ (self):
        #define some default values
        data = np.empty(3,2)
        data.fill(np.nan)
        self.properties_mephisto = pd.DataFrame(data, columns=['k','tau'])
        
        data = np.empty(6,3)
        data.fill(np.nan)        
        self.nonlin_correction = pd.DataFrame(data, columns=['coeff_x','coeff_y', 'coeff_z'])
        
        self.pedestalcorr = np.array([np.nan, np.nan, np.nan])
        
        data = np.empty(22,3)
        data.fill(np.nan)
        self.focus_shift = pd.DataFrame(data, columns=['xmean', 'ymean', 'zmean'])
        self.focus_shift.index.name = 'voltage'
        self.focus_shift = self.focus_shift.to_xarray()

        data = np.empty(3,2)
        data.fill(np.nan)
        self.Flyvbjerg_Corr = pd.DataFrame(data, columns=['Flyvbjerg Coef','Flyvbjerg Corr'])

        self.coverslip_corr = np.nan

        self.offsets = np.array([np.nan, np.nan, np.nan])
         	

    def load_Mephisto_properties(self, path_to_springconst):
        '''loads the spring constants and autocorrelation times of the Mephisto trap into a pandas dataframe
        
        Args:
            path_to_springconst (string): path to the file containing the Mephisto spring constants
        '''
        self.properties_mephisto = pd.read_csv(path_to_springconst, sep='\t',header=0, names=['k', 'tau'],skiprows=0)
    
    def load_nonlin_correction(self, path_to_nonlincorrection):
        '''loads the nonlinear correction coefficients into a pandas dataframe.
        
        Args:
            path_to_nonlincorrection (string): path to the file containing the coefficients.
        '''
        self.nonlin_correction = pd.read_csv(path_to_nonlincorrection, sep='\t',header=0, names=['x', 'y', 'z'],skiprows=0)
    
    def load_pedestal_correction(self, path_to_initialStagePos, path_to_OffsetsX, path_to_OffsetsY, path_to_OffsetsZ, path_to_OffsetStagePos, path_to_ZeroOffsets):
        '''loads all files necessary to correct for light scattered by the pedestal beads, and computes a matrix of correction voltages.
        
        Args:
            path_to_initialStagePos (string): path to stagePosAtStartofFeedback.csv, the file containing the stage position at the start of the active feedback loop that stabilizes the sample.
            path_to_OffsetsX (string): path to OffsetsX.csv, containing the offset voltages on the X channel of the detector introduced by the pedestal bead. This is a 2D matrix; each point in the matrix represents one scanned position in the sample.
            path_to_OffsetsY (string): path to OffsetsY.csv, containing the offset voltages on the Y channel of the detector introduced by the pedestal bead. This is a 2D matrix; each point in the matrix represents one scanned position in the sample.
            path_to_OffsetsZ (string): path to OffsetsZ.csv, containing the offset voltages on the Z channel of the detector introduced by the pedestal bead. This is a 2D matrix; each point in the matrix represents one scanned position in the sample.
            path_to_OffsetStagePos (string): path to OffsetStagePos.csv, containing the stage positions during the scan of the offset voltages.
            path_to_ZeroOffsets (string): path to ZeroOffsetsPedestal.csv, containing the x, y and z voltages on the detector for a position far from a pedestal bead.
        '''
        initial_stagepos = np.genfromtxt(path_to_initialStagePos) * 1e-6
        
        offset_stagepos = np.genfromtxt(path_to_OffsetStagePos) * 1e-6
        StartX = offset_stagepos[0][0]
        StartY = offset_stagepos[0][1]
        dx = (offset_stagepos[1][0]-offset_stagepos[0][0])
        dy = (offset_stagepos[1][1]-offset_stagepos[0][1])

        zero_offsets = np.genfromtxt(path_to_ZeroOffsets) 
        
        dataX = np.genfromtxt(path_to_OffsetsX) - zero_offsets[0] #zero the offset data
        dataY = np.genfromtxt(path_to_OffsetsY) - zero_offsets[1]
        dataZ = np.genfromtxt(path_to_OffsetsZ) - zero_offsets[2]

        zero_offsets = np.genfromtxt(path_to_ZeroOffsets) 

        OffsetsX = xr.DataArray(dataX,
                       dims=('y', 'x'),
                       coords={'x': StartX + np.arange(dataX.shape[1]) * dx, 'y': StartY + np.arange(dataX.shape[0]) * dy})
        OffsetsY = xr.DataArray(dataY,
                       dims=('y', 'x'),
                       coords={'x': StartX + np.arange(dataY.shape[1]) * dx, 'y': StartY + np.arange(dataY.shape[0]) * dy})
        OffsetsZ = xr.DataArray(dataZ,
                       dims=('y', 'x'),
                       coords={'x': StartX + np.arange(dataZ.shape[1]) * dx, 'y': StartY + np.arange(dataZ.shape[0]) * dy})
        
        self.pedestalcorr = [OffsetsX.interp(x = initial_stagepos[0], y = initial_stagepos[1]), \
            OffsetsY.interp(x = initial_stagepos[0], y = initial_stagepos[1]), \
                OffsetsZ.interp(x = initial_stagepos[0], y = initial_stagepos[1])]
        
    def load_focusshift(self, path_to_focusshift):
        '''loads the focus shift of the 852nm laser vs its intensity setpoint into an xarray DataSet.
        
        Args:
            path_to_focusshift (string): path to 852_focus_shift.csv which contains the focus shift data.
        '''
        self.focus_shift = pd.read_csv(path_to_focusshift, sep='\t', index_col=0).to_xarray()
        
    def load_FlyvbjergCorr(self, path_to_flyvbjergCorr):
        '''loads the Flyvbjerg correction coefficients.

        Args:
            path_to_flyvbjergCorr (string): path to FlyvbjergCorrection.csv containing the correction coefficients.
        '''

        self.Flyvbjerg_Corr = pd.read_csv(path_to_flyvbjergCorr, sep='\t')

    def load_correctionCloseToCoverslip(self, path_to_852Pos, path_to_Extended852Pos):
        '''Close to the coverslip the position sensor has a ~10% systematic error. Load two position signals that are nominally 100 nm apart and compute a correction factor.
        
        Args: 
            path_to_852Pos (string): path to 852ZeroPosBeforeTrials.csv
            path_to_Extended852Pos (string): path to 852ExtendedPosBeforeTrials.csv
        '''

        zero = np.genfromtxt(path_to_852Pos)
        extended = np.genfromtxt(path_to_Extended852Pos)
        self.coverslip_corr = (extended[1] - zero[1])/100.0

    def load_offsets(self, path_to_offsets):
        '''load offsets to set the middle of the optical trap as the origin (0,0,0). Offsets are to be added post-calibration to the position traces.

        Args:
            path_to_offsets (string): path to offsets.csv.
        '''
        self.offsets = np.genfromtxt(path_to_offsets)

    def loadAndCalibrateSignals(self, path_to_signals):
        '''load the signals, calibrate, and correct them.

        Args: 
            path_to_signals (string): path to the signals.tdms file.
        '''

        tdms_file = nptdms.TdmsFile("signals.tdms")
        xchannel = tdms_file.object('position_data', 'xwaveFSD')
        ychannel = tdms_file.object('position_data', 'ywaveFSD')
        zchannel = tdms_file.object('position_data', 'zwaveFSD')
        xdata = xchannel.data

        

    def _calibrateAndCorrect_Channel(self, channel, direction):
        '''apply calibration and corrections to one channel of the signals.

        Args:
            channel (nptdms.channel): channel object containing the time series.
            direction (int): 0, 1, or 2 for the x, y, or z direction.

        Returns: wave (xarray.DataArray):  calibrated and corrected data for the chosen channel.
        '''

        strdir = ['x', 'y', 'z']
        
        #correct for influence of the pedestal
        data = np.array(channel.data)
        data = np.subtract(data, self.pedestalcorr[direction])

        d = direction
        coef = np.array(self.nonlin_correction[strdir[direction]])
        #calibrate and correct for nonlinear detector
        datacal = np.polynomial.polynomial.polyval(data, coef, tensor=False)
        datacal = np.multiply(datacal, 1e-9) #result is in nm; convert to meters to stay in SI units.

        #apply the offsets
        datacal = np.subtract(datacal, self.offsets[direction])

        #apply Flyvbjerg correction
        datacal = np.divide(datacal, self.Flyvbjerg_Corr['Flyvbjerg Corr'][direction])
        
        #apply coverslip correction to x and y channel. We do not know if (or how much) the z detector is rescaled, but we also do not pull along z, so who cares.
        if(direction < 3):
            datacal = np.divide(datacal, self.coverslip_corr)

        #we are done, build a nice xarray to return to the user
        time = channel.time_track()
        wave = xr.DataArray(datacal,
                    dims='time',
                    coords = {'time': time})

        return wave

