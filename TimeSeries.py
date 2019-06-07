import pandas as pd
import numpy as np
import xarray as xr
import nptdms
import plotly.graph_objs as go
from scipy.optimize import curve_fit, leastsq
from scipy.stats import ttest_ind
from scipy.signal import savgol_filter
import holoviews as hv
import holoviews.operation.datashader as hd
hd.shade.cmap=["lightblue", "darkblue"]
import datashader as ds
from single_molecule_mechanics.ProteinModels import xSeriesWLCe
import warnings
from collections import OrderedDict 
from multiprocessing import Pool


class TimeSeriesLoader():
    '''Provides data structures and methods to analyze single-molecule data.'''

    def __init__ (self):
        #define some default values
        data = np.empty((3,2))
        data.fill(np.nan)
        self.properties_mephisto = pd.DataFrame(data, columns=['k','tau'])
        
        data = np.empty((6,3))
        data.fill(np.nan)        
        self.nonlin_correction = pd.DataFrame(data, columns=['coeff_x','coeff_y', 'coeff_z'])
        
        self.pedestalcorr = np.array([np.nan, np.nan, np.nan])
        
        data = np.empty((22,3))
        data.fill(np.nan)
        self.focus_shift = pd.DataFrame(data, columns=['xmean', 'ymean', 'zmean'])
        self.focus_shift.index.name = 'voltage'
        self.focus_shift = self.focus_shift.to_xarray()

        data = np.empty((3,2))
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
                       coords={'x': StartX + np.arange(dataX.shape[1]) * dx, 'y': StartY + np.arange(dataX.shape[0]) * dy},
                       name='Offsets X')
        OffsetsY = xr.DataArray(dataY,
                       dims=('y', 'x'),
                       coords={'x': StartX + np.arange(dataY.shape[1]) * dx, 'y': StartY + np.arange(dataY.shape[0]) * dy},
                       name='Offsets Y')
        OffsetsZ = xr.DataArray(dataZ,
                       dims=('y', 'x'),
                       coords={'x': StartX + np.arange(dataZ.shape[1]) * dx, 'y': StartY + np.arange(dataZ.shape[0]) * dy},
                       name='Offsets Z')
        
        self.pedestalcorr = np.array([OffsetsX.interp(x = initial_stagepos[0], y = initial_stagepos[1]), \
            OffsetsY.interp(x = initial_stagepos[0], y = initial_stagepos[1]), \
                OffsetsZ.interp(x = initial_stagepos[0], y = initial_stagepos[1])])
        
    def load_focusshift(self, path_to_focusshift):
        '''loads the focus shift of the 852nm laser vs its intensity setpoint into an xarray DataSet.
        
        Args:
            path_to_focusshift (string): path to 852_focus_shift.csv which contains the focus shift data.
        '''

        #this needs to be a DataArray so that we can easily interpolate.
        #Drop the first datapoint (power less than 0.5). When the laser is off we cannot know where the focus has shifted to, so that value is meaningless.
        self.focus_shift = pd.read_csv(path_to_focusshift, sep='\t', index_col=0)[0.5:].to_xarray() * 1e-9
        
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
        self.offsets = np.genfromtxt(path_to_offsets, skip_header=1) *1e-9

    def loadAndCalibrateSignals(self, path_to_signals):
        '''load the signals, calibrate, and correct them.

        Args: 
            path_to_signals (string): path to the signals.tdms file.
        '''

        tdms_file = nptdms.TdmsFile("signals.tdms")
        self.xchannel = tdms_file.object('position_data', 'xwaveFSD')
        self.ychannel = tdms_file.object('position_data', 'ywaveFSD')
        self.zchannel = tdms_file.object('position_data', 'zwaveFSD')
        power = tdms_file.object('position_data', 'powerFSD')

        self.powerArray = xr.DataArray(np.array(power.data),
                                        dims = 'time',
                                        coords = {'time': power.time_track()},
                                        name='power')
        
        self.data_calibrated = xr.Dataset({'x': self._calibrateAndCorrect_Channel(self.xchannel, 0), \
            'y': self._calibrateAndCorrect_Channel(self.ychannel, 1), \
                'z': self._calibrateAndCorrect_Channel(self.zchannel, 2),})

    def load_dkdI(self, path_to_dkdI):
        '''load the slopes of the stiffness-intensity curves for the 852nm laser

        Args: 
            path_to_dkdI (string): path to 852_k_sens_dk_dI.csv.
        '''
        self.dkdI = np.genfromtxt(path_to_dkdI)

    def computeForceArray(self, direction, displacement=200e-9):
        '''computes the force xarray.DataArray from the probe position and stimulus intensity
        
        Args:
            direction: 0, 1, or 2 for the x, y, or z direction of pulling.
            displacement (float): distance between the center of the 852nm trap and the mephisto trap.
        
        '''
        springconst = np.multiply(self.powerArray, self.dkdI[direction])
        if(direction != 1):
            print("pulling direction x or z has not yet been implemented")
            pass
        
        #compute the focus shift based on the powerArray
        focusshiftArray = self.focus_shift['ymean'].interp(voltage = self.powerArray, kwargs={'fill_value': 'extrapolate'})
    
        self.forceArray = springconst * (displacement-self.data_calibrated['y']+focusshiftArray)


    def _calibrateAndCorrect_Channel(self, channel, direction):
        '''apply calibration and corrections to one channel of the signals.

        Args:
            channel (nptdms.channel): channel object containing the time series.
            direction (int): 0, 1, or 2 for the x, y, or z direction.

        Returns: wave (xarray.DataArray):  calibrated and corrected data for the chosen channel.
        '''

        strdir = ['x', 'y', 'z']
        
        #correct for influence of the pedestal
        if(np.isfinite(self.pedestalcorr[direction]) is False):
            raise ValueError('pedestal corr is not finite')
        
        data = np.array(channel.data)
        data = np.subtract(data, self.pedestalcorr[direction])

        d = direction
        coef = np.array(self.nonlin_correction[strdir[direction]])
        #calibrate and correct for nonlinear detector
        if(np.isfinite(coef) is False):
            raise ValueError('nonlin coef is not finite')
        datacal = np.polynomial.polynomial.polyval(data, coef, tensor=False)
        datacal = np.multiply(datacal, 1e-9) #result is in nm; convert to meters to stay in SI units.

        #apply the offsets
        if(np.isfinite(self.offsets[direction]) is False):
            raise ValueError('offsets are not finite')
        datacal = np.subtract(datacal, self.offsets[direction])

        #apply Flyvbjerg correction
        if(np.isfinite(self.Flyvbjerg_Corr['Flyvbjerg Corr'][direction]) is False):
            raise ValueError('Flyvbjerg corr is not finite')
        datacal = np.divide(datacal, self.Flyvbjerg_Corr['Flyvbjerg Corr'][direction])
        
        #apply coverslip correction to x and y channel. We do not know if (or how much) the z detector is rescaled, but we also do not pull along z, so who cares.
        if(np.isfinite(self.coverslip_corr) is False):
            raise ValueError('coverslip corr is not finite')
        if(direction < 3):
            datacal = np.divide(datacal, self.coverslip_corr)

        #we are done, build a nice xarray to return to the user
        time = channel.time_track()
        wave = xr.DataArray(datacal,
                    dims='time',
                    coords = {'time': time},
                    name=strdir[direction] + 'position')

        return wave

class Histogram3D():
    '''3D histogram from an x, y, z time series'''

    def __init__(self, timeseries, numbins, binsize):
        '''Construct a 3D histogram from an x, y, z time series
         
        Args:
           timeseries (np.array): (3 x m)-dimensional array (time series of length m, 3 dimensions)
           numbins (int): number of bin edges along each axis
           binsize (float): bin size in meters
        '''
        start = (-1) * numbins/2 * binsize
        end = numbins/2 * binsize
        bins_oneaxis = np.arange(start, end, binsize)
        bins = [bins_oneaxis, bins_oneaxis, bins_oneaxis]
        self.histo3D = np.histogramdd(sample=(timeseries[0], timeseries[1], timeseries[2]), bins=bins)
        self._GlobalEnergyMinimum = np.array([0.0, 0.0, 0.0, 0.0]) #location and value -> N = 4

    def plotlyIsosurface(self, isomin, isomax, title):
        '''Return a plotly FigureWidget containing an isosurface representation of the 3D histogram
        
        Args: 
            isomin (int): minimum value of isosurface colorscale
            isomax (int): max value of isosurface colorscale
            title (string): title of the plot
        '''

        # generate X, Y, Z, Values arrays. Plotly needs these as inputs to the Isosurface function. We are storing these as properties
        # so that the user could access them to plot more involved isosurface representations from the Jupyter notebook.
        self.histo3D_unrolled = np.reshape(self.histo3D[0], [-1])
        self.histo3D_unrolled_coords = [ [x,y,z] for z in self.histo3D[1][2][:-1] for y in self.histo3D[1][1][:-1] for x in self.histo3D[1][0][:-1]]

        data = [go.Isosurface(
            x=np.array(self.histo3D_unrolled_coords)[:,0],
            y=np.array(self.histo3D_unrolled_coords)[:,1],
            z=np.array(self.histo3D_unrolled_coords)[:,2],
            value=self.histo3D_unrolled,
            isomin=isomin,
            isomax=isomax,
            colorscale = 'Blues'
        )]
        layout = go.Layout(title=title)
        return go.FigureWidget(data, layout)

    def surfaceOfMaxOccupancy(self, perpendicular_axis):
        '''Return an xarray.DataArray of the surface of maximum occupancy. Starts searching for this surface perpendicular to 'perpendicular_axis'

        Args: 
            perpendicular_axis (int): 0, 1, or 2 for x, y, or z.
        '''        

        #we will fit a Gaussian to each column of voxels along the perpendicular_axis, with the guesses from surfaceposguess and surfacewidthguess
        fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
        errfunc  = lambda p, x, y: (y - fitfunc(p, x))

        histdata = self.histo3D[0]
        histedge = self.histo3D[1][perpendicular_axis]
        hist_bin_centers = histedge[0:-1] + (histedge[1]-histedge[0])/2

        #iterate over xy positions. we will call the direction of perpendicular_axis the z axis.
        #rearrange the 3D matrix to orient it appropriately:
        order = np.array([0, 1, 2]) - perpendicular_axis
        histdata_rearr = histdata.transpose(*order)
        surfaceposguess = histdata_rearr.argmax(axis=2)
        surfaceposguess = hist_bin_centers[surfaceposguess]
        surfacemaxguess = np.max(histdata_rearr, axis=2)
        surfacewidthguess = np.std(histdata_rearr, axis=2)
        counts = np.sum(histdata_rearr, axis=2)
        
        surfacepos = np.copy(surfaceposguess.astype(float))
        
        for index, pos in np.ndenumerate(surfaceposguess):
            if(counts[index] > 500):
                vals = histdata_rearr[index]
                init = [surfacemaxguess[index], pos, surfacewidthguess[index]]
                out = leastsq(errfunc, init, args=(hist_bin_centers, vals))
                surfacepos[index] = out[0][1]
                
                #update the global energy minimum value and location
                if(surfacemaxguess[index] > self.GlobalEnergyMinimum[3]):
                    self.GlobalEnergyMinimum[0] = hist_bin_centers[index[0]]
                    self.GlobalEnergyMinimum[1] = hist_bin_centers[index[1]]
                    self.GlobalEnergyMinimum[2] = out[0][1]
                    self.GlobalEnergyMinimum[3] = surfacemaxguess[index]
            else:
                surfacepos[index] = 0

        self.maxOccupancySurface = xr.DataArray(surfacepos,
                                                dims = ['x', 'y'],
                                                coords = {'x': hist_bin_centers, 'y': hist_bin_centers},
                                                name='surface of maximum occupancy')

        return self.maxOccupancySurface
    
    def EnergyLandscapeAlongTether(self, dx, numpnts, axis=1):
        '''Return the energy landscape along the tether direction. Determines the orientation of the tether from the orientation of the bead fluctuations.
        
        Args: 
            dx (float): stepsize in meters (choose this similar to the binsize of the histogram).
            numpnts (int): number of steps along the tether direction. The global energt minimum will be at the center of the steps.
            axis (int): the axis with which the tether is approximately aligned.
        '''

        _ = self.surfaceOfMaxOccupancy(axis)
        surfnorm = self.SurfaceNormalAtGlobMinimum
        
        startpos = np.array([self.GlobalEnergyMinimum[0:3]] - surfnorm * dx * numpnts/2 )
        path = self.SurfaceNormalAtGlobMinimum[:, np.newaxis] * dx * np.array( [np.arange(numpnts), np.arange(numpnts), np.arange(numpnts)]) + startpos.T
        intpArray = self.histo3Darray_transposed(axis).interp(x=path[0,:], y=path[1,:], z=path[2,:])
        occupancy = [intpArray.values[i, i, i] for i in range(numpnts)]
        
        energy = (-1)* np.log(occupancy)

        energyArray = xr.DataArray(energy,
                                    dims = 'tether direction',
                                    coords = {'tether direction': np.arange(numpnts) * dx},
                                    attrs = {'coord3D': path},
                                    name='energy')
        occupancyArray = xr.DataArray(occupancy,
                                    dims = 'tether direction',
                                    coords = {'tether direction': np.arange(numpnts) * dx},
                                    attrs = {'coord3D': path},
                                    name='occupancy')
        return xr.Dataset({'occupancy': occupancyArray, 'energy': energyArray})

    def ProteinAnchorPosition(self, energylandscape=None):
        '''Return the coordinates of the anchor position of the protein

        Args:
            energy landscape (xarray.DataArray): energy landscape of the tethered protein. Must contain the three dimensional coordinates of the values as 'coord3D' attribute.
                                                If you do not pass an energylandscape, this function will attempt to compute it for you.
        '''
        if energylandscape == None:
            energylandscape = self.EnergyLandscapeAlongTether(dx=2e-9, numpnts=300, axis=1)['energy']

        cutoff = int(np.floor(len(energylandscape)/2))
        anchorindex = np.max(np.where(energylandscape[0:cutoff]==np.inf)) #index in the energy landscape whose value is inf while being closest to the energy minimum
        path = energylandscape.attrs['coord3D']
        anchorpos = path[:,int(anchorindex)]
        return anchorpos

    @property
    def histo3DArray(self):
        '''Return the 3D histogram object as an xarray.'''
        histdata = self.histo3D[0]
        histedge = self.histo3D[1][1]
        hist_bin_centers = histedge[0:-1] + (histedge[1]-histedge[0])/2
        return xr.DataArray(histdata,
                            dims = ['x', 'y', 'z'],
                            coords = {'x': hist_bin_centers, 'y': hist_bin_centers, 'z': hist_bin_centers},
                            name='3D histogram')

    def histo3Darray_transposed(self, pullingaxis=1):
        '''Return the 3D histogram object as an xarray. Rotate the array so that the pulling direction alignes with the z axis
        
        Args:
            pullingaxis (int): direction in which the tether will be pulled. x=0, y=1, z=2.
        '''
        order = np.array([0, 1, 2]) - pullingaxis
        histdata = self.histo3D[0]
        histdata_rearr = histdata.transpose(*order)
        histedge = self.histo3D[1][1]
        hist_bin_centers = histedge[0:-1] + (histedge[1]-histedge[0])/2
        return xr.DataArray(histdata_rearr,
                            dims = ['x', 'y', 'z'],
                            coords = {'x': hist_bin_centers, 'y': hist_bin_centers, 'z': hist_bin_centers})

    @property
    def GlobalEnergyMinimum(self):
        '''Return the position of the most highly occupied point in the histogram. In the current implementation this is given in the coordinate system in which z is the pulling direction!
        '''
        return self._GlobalEnergyMinimum

    @property
    def SurfaceNormalAtGlobMinimum(self):
        '''Return the surface normal of the maxOccupancySurface at the GlobalEnergyMinimum'''
        
        #find tangent in x direction
        xmin = self.GlobalEnergyMinimum[0]
        ymin = self.GlobalEnergyMinimum[1]

        tangentx = np.zeros(3)
        tangentx[0]=20e-9
        tangentx[1]=0
        tangentx[2]=self.maxOccupancySurface.interp(x=xmin+10e-9, y=ymin)-self.maxOccupancySurface.interp(x=xmin-10e-9, y=ymin)
        
        tangenty = np.zeros(3)
        tangenty[0]=0
        tangenty[1]=20e-9
        tangenty[2]=self.maxOccupancySurface.interp(x=xmin, y=ymin+10e-9)-self.maxOccupancySurface.interp(x=xmin, y=ymin-10e-9)

        surfnormal = np.zeros(3)
        surfnormal[0]=tangentx[1]*tangenty[2]-tangentx[2]*tangenty[1]
        surfnormal[1]=-tangentx[0]*tangenty[2]+tangentx[2]*tangenty[0]
        surfnormal[2]=tangentx[0]*tangenty[1]-tangentx[1]*tangenty[0]
        
        surfnormalnorm=np.sqrt(surfnormal[0]**2+surfnormal[1]**2+surfnormal[2]**2)
        surfnormal[0]/=surfnormalnorm
        surfnormal[1]/=surfnormalnorm
        surfnormal[2]/=surfnormalnorm

        return surfnormal

class Histogram2D():
    '''represents a 2D histogram constructed from two time series'''

    def __init__(self, wave1, wave2, delta1, delta2, min1=None, max1=None, min2=None, max2=None):
        '''Construct a new 2D histogram.
        
        Args:
            wave1 (xarray.Dataarray): first time series.
            wave2 (xarray.Dataarray): second time series.
            delta1 (float): binsize of bins along the axis of timeseries 1.
            delta2 (float): binsize of bins along the axis of timeseries 2.
            min1 (float): minimum value of time series 1 to be included in histogram. If None: include the smallest value.
            max1 (float): maximum value of time series 1 to be included in histogram. If None: include the largest value.
            min2 (float): minimum value of time series 2 to be included in histogram. If None: include the smallest value.
            max2 (float): maximum value of time series 2 to be included in histogram. If None: include the largest value.
        '''
        if min1 is None:
            min1 = np.min(wave1.values)
        if min2 is None:
            min2 = np.min(wave2.values)  
        if max1 is None:
            max1 = np.max(wave1.values)
        if max2 is None:
            max2 = np.max(wave2.values)       

        bins1 = np.arange(min1, max1, delta1)
        bins2 = np.arange(min2, max2, delta2)
        bins = [bins1, bins2]
        histo2Dvals = np.histogramdd(sample=(wave1.values, wave2.values), bins=bins)
        histdata = histo2Dvals[0]
        histedge1 = histo2Dvals[1][0]
        histedge2 = histo2Dvals[1][1]
        hist_bin_centers_1 = histedge1[0:-1] + (histedge1[1]-histedge1[0])/2
        hist_bin_centers_2 = histedge2[0:-1] + (histedge2[1]-histedge2[0])/2
        self.histo2D = xr.DataArray(histdata,
                            dims = [wave1.name, wave2.name],
                            coords = { wave2.name: hist_bin_centers_2, wave1.name: hist_bin_centers_1})
        


class ForceExtHeatmap(Histogram2D):
    '''represents a 2D histogram of force-extension trials.'''

    def __init__(self, forcewave, extensionwave, maxforce, maxextension, dforce, dextension):
        '''construct a new force-extension heatmap
        
        Args: 
            forcewave (xarray.Dataarray): force time trace
            extensionwave(xarray.Dataarray): extension time trace
            maxforce (float): maximum force
            maxextension (float): maximum extension
            dforce (float): force bin width
            dextension (float): extension bin width
        '''
        super().__init__(forcewave, extensionwave, delta1 = dforce, delta2 = dextension, min1=0, max1 = maxforce, min2 = 0, max2=maxextension)

    @property
    def heatmap(self):
        return self.histo2D


class ForceRampHandler(object):
    '''Provides tools to handle single-molecule force ramp data.'''

    def __init__(self, forcewave, extensionwave, numtrials, smooth = True, savgol_coeff = [101,3]):
        '''Create a new ForceRampHandler.

        Args:
            forcewave (xarray.Dataarray): an appropriately cut force time trace (must contain only ramps, nothing else).
            extensionwave (xarray.Dataarray): an appropriately cut extension time trace (over the same time domain as the forcewave).
            numtrials (int): number of ramps in the time trace
            smooth (Boolean): smooth the data using a Savitzky Golay filter?
            savgol_coeff (list): [savgol window size, polynomial order]
        '''
        #smooth?
        if(smooth):
            extensionwave = savgol_filter(extensionwave, savgol_coeff[0],savgol_coeff[1])
            forcewave = savgol_filter(forcewave, savgol_coeff[0],savgol_coeff[1])  
        
        #stack each individual phase of each cycle into its own row of a matrix
        length = int(np.floor(len(np.array(forcewave))/(numtrials*2)))
        fwave = forcewave[0:numtrials*2*length]
        exwave = extensionwave[0:numtrials*2*length]
        forcematrix = np.array(fwave).reshape((-1, length))
        extensionmatrix = np.array(exwave).reshape((-1, length))
        force_pulls = forcematrix[::2]
        force_releases = forcematrix[1::2]
        extension_pulls = extensionmatrix[::2]
        extension_releases = extensionmatrix[1::2]

        self.pulls = np.stack((extension_pulls, force_pulls), axis=2)
        self.releases = np.stack((extension_releases, force_releases), axis=2)

    @property
    def pullsXr(self):
        '''Return a list of xarray.DataArray objects for all pulls'''
        pulls = []
        for pull in self.pulls:
            pullArray = xr.DataArray(pull[:,1],
                                    dims=['extension'],
                                    coords = {'extension': pull[:,0]},
                                    name='force')
            pulls.append(pullArray)
        return pulls

    @property
    def releasesXr(self):
        '''Return a list of xarray.DataArray objects for all pulls'''
        releases = []
        for release in self.releases:
            releaseArray = xr.DataArray(release[:,1],
                                    dims=['extension'],
                                    coords = {'extension': release[:,0]},
                                    name='force')
            releases.append(releaseArray)
        return releases

    def LayoutOfCycles(self, start, stop, fits_pulls = None, fits_releases = None):
        '''Return a holoviews layout of force-extension cycles.

        Args:
            start (int): number of first cycle in layout.
            stop (int): number of last cycle in layout.
            fits_pulls (list of lists of xr.DataArray): outer list: cycle, inner list: segment within cycle.
            fits_releases (list of lists of xr.DataArray): outer list: cycle, inner list: segment within cycle.
        Returns:
            layout (holoviews/matplotlib): layout of cycles.
        '''

        all_cycles = hv.Layout()
        for cyclenum in np.arange(start, stop):
            pull = self.pullsXr[cyclenum]
            release = self.releasesXr[cyclenum]
            
            currentcycle = OrderedDict()
            currentcycle['extension'] = hv.Curve(pull).opts(color='red')
            currentcycle['relaxation'] = hv.Curve(release).opts(color='blue')
            
            if (fits_pulls is not None and fits_releases is not None):
                for i, pullfit in enumerate(fits_pulls[cyclenum]):
                    currentcycle['fit (pull) ' + str(i)] = hv.Curve(pullfit).opts(color='black', linestyle='dashed')
                for i, releasefit in enumerate(fits_releases[cyclenum]):
                    currentcycle['fit (relaxation) ' + str(i)] = hv.Curve(releasefit).opts(color='black', linestyle='dashed')

            layout = hv.NdOverlay(currentcycle, kdims='ramp', sort=False).options({'Curve': {'xlim': (0, 200e-9), 'ylim': (0, 70e-12)}})
            layout.opts(show_legend = False)
            #cyclelayout = hd.dynspread(hd.datashade(layout, aggregator=ds.count_cat('k'))).opts(xrotation=0, xlim=(0, 200e-9), ylim=(0, 70e-12), xformatter='%.1e',yformatter='%.1e',xlabel='extension (m)', ylabel='force (N)')
            cyclelayout = layout.opts(xlabel='extension (m)', ylabel='force (N)')
            all_cycles += cyclelayout          

        hd.shade.color_key=None #reset
        hv.extension('matplotlib')
        return all_cycles.cols(4)

    def fitAllPullsWithWLCs(self, pulls, numsdevs=3, force_threshold=3e-12):

        lcs_all_pulls = []
        lps_all_pulls = []
        Ks_all_pulls = []
        dLc_vs_F = []
        fit_pulls = []

        for (i, pull) in enumerate(pulls):
            print(i)
            (steps_start, steps_end) = self._detectConfChange(pull[:,1], pull[:,0], pull=True, numsdevs=numsdevs, force_threshold=force_threshold, window=1000)
            segments = self._confChangeToSegments(steps_start, steps_end, pull[:,1], pull[:,0])
            lcs_one_pull = []
            lps_one_pull = []
            Ks_one_pull = []
            Fmaxs_one_pull = [] #maximum force at the end of each segment (i.e. the forces at which a rip happened)
            segfit_one_pull =[]
            for seg in segments:
                (params, params_cov, seg_fit, fitfailed) = self._fitSeriesWLCs(seg, lps = [0.5e-9, 4e-9], lcs = [37e-9, 50e-9], Ks=[7.2e-3*37e-9, 7.2e-3*37e-9], holdconst= [True, False, True, False, True, False])
                if (fitfailed): #curve fit failed
                    continue
                lcs_one_pull.append(params[3])
                lps_one_pull.append(params[1])
                Ks_one_pull.append(params[5])
                Fmaxs_one_pull.append(np.max(seg.values))
                segfit_one_pull.append(seg_fit)
            if(len(lcs_one_pull) > 1):
                #conformational change detected
                dLcs = np.diff(lcs_one_pull)
                for dLc, F in zip(dLcs, Fmaxs_one_pull):
                    dLc_vs_F.append([F, dLc, i])
            lcs_all_pulls.append(lcs_one_pull)
            lps_all_pulls.append(lps_one_pull)
            Ks_all_pulls.append(Ks_one_pull)
            fit_pulls.append(segfit_one_pull)
        
        return (dLc_vs_F, lcs_all_pulls, lps_all_pulls, Ks_all_pulls, fit_pulls)

    def fitAllPullsWithWLCs_parallel(self, numprocesses, pulls, numsdevs=3, force_threshold=3e-12):

        with Pool(numprocesses) as p:
            (lcs_all_pulls, lps_all_pulls, Ks_all_pulls, dLc_vs_F_all_pulls, fit_pulls) = p.map(_processOnePull, pulls)
            
        return (lcs_all_pulls, lps_all_pulls, Ks_all_pulls, dLc_vs_F_all_pulls, fit_pulls)

    def _processOnePull(self, pull, numsdevs=3, force_threshold=3e-12):
        (steps_start, steps_end) = self._detectConfChange(pull[:,1], pull[:,0], pull=True, numsdevs=numsdevs, force_threshold=force_threshold, window=1000)
        segments = self._confChangeToSegments(steps_start, steps_end, pull[:,1], pull[:,0])
        lcs_one_pull = []
        lps_one_pull = []
        Ks_one_pull = []
        dLc_vs_F_one_pull = []
        Fmaxs_one_pull = [] #maximum force at the end of each segment (i.e. the forces at which a rip happened)
        segfit_one_pull =[]
        for seg in segments:
            (params, params_cov, seg_fit, fitfailed) = self._fitSeriesWLCs(seg, lps = [0.5e-9, 4e-9], lcs = [37e-9, 50e-9], Ks=[7.2e-3*37e-9, 7.2e-3*37e-9], holdconst= [True, False, True, False, True, False])
            if (fitfailed): #curve fit failed
                continue
            lcs_one_pull.append(params[3])
            lps_one_pull.append(params[1])
            Ks_one_pull.append(params[5])
            Fmaxs_one_pull.append(np.max(seg.values))
            segfit_one_pull.append(seg_fit)
        if(len(lcs_one_pull) > 1):
            #conformational change detected
            dLcs = np.diff(lcs_one_pull)
            for dLc, F in zip(dLcs, Fmaxs_one_pull):
                dLc_vs_F_one_pull.append([F, dLc])
        
        return (lcs_one_pull, lps_one_pull, Ks_one_pull, dLc_vs_F_one_pull, segfit_one_pull)


    def _detectConfChange(self, forcewave, extensionwave, pull=True, numsdevs=3, force_threshold=3e-12, window=1000):
        '''Runs a statistical test to determine conformational changes in the extension wave. Only considers data for which the force is larger than force_threshold.

            Args:
                forcewave (np.array): force time trace
                extensionwave (np.array): extension time trace
                pull (boolean): Is the provided data a pull or a relaxation?
                numsdevs (int): identify events if they are this many number of sdevs above noise
                force_threshold (float): threshold for the force: do not test data with a force smaller than this value 
                window (int): number of data points to consider for the statistical test.
        '''
        if(pull is False): #create waves with monotonically increasing force
            forcewave = np.flip(forcewave)
            extensionwave = np.flip(extensionwave)
        
        mask = forcewave > force_threshold
        detected_steps = np.copy(forcewave)
        detected_steps.fill(0)
        
        #where does the ramp start?
        startindex = np.argmax(mask)

        for index in np.arange(startindex + window, len(extensionwave) - window, 1):
            mean_before = np.mean(extensionwave[index-window:index])
            mean_after =  np.mean(extensionwave[index:index+window])
            sdev_before = np.std(extensionwave[index-window:index])
            sdev_after = np.std(extensionwave[index:index+window])
            detected_steps[index] = np.abs(mean_after-mean_before) > numsdevs*(sdev_before+sdev_after)/2
        
        #undo the flip:
        if(pull is False):
            detected_steps = np.flip(detected_steps)

        #find beginning and end of each transition
        d_steps = np.diff(detected_steps)
        steps_start = np.where(d_steps==1)
        steps_end = np.where(d_steps==-1)
        return (steps_start, steps_end)

    def _confChangeToSegments(self, steps_start, steps_end, forcewave, extensionwave):
        '''Return a segmentated representation fo forcewave and extension wave, deliminated by the conformational changes

            Args:
                steps_start (np.array): detected start indices of conformational changes, output from _detectConfChange
                steps_end (np.array): detected end indices of conformational changes, output from _detectConfChange
                forcewave (np.array): force time trace
                extensionwave (np.array): extension time trace
            
            Returns:
                segments (list): list of xr.DataArray containing the force extension segments.
        '''
        _steps_start = np.insert(steps_start, 0, 0) #insert a zero at the beginning of _steps_start
        _steps_end = np.insert(steps_end, 0, 0)
        
        _steps_start = np.append(_steps_start, len(forcewave))
        _steps_end = np.append(_steps_end, len(forcewave))

        segments = []
        for start, end in zip(_steps_start[1:], _steps_end[0:-1]):
            force_segment = forcewave[end:start]
            extension_segment = extensionwave[end:start]
            seg = xr.DataArray(force_segment,
                                dims=['extension'],
                                coords={'extension': extension_segment},
                                name='force')
            segments.append(seg)

        return segments

    def _fitSeriesWLCs(self, forceExtension, lps, lcs, Ks, holdconst):
        '''Fit force-extension data with a series of WLCs.

        Args:
            forceExtension (xr.xarray): force extension as an xarray (values: 'force', coords: 'extension')
            lps (list of float): persistence lengths
            lcs (list of float): contour lengths
            Ks (list of float): enthalpic moduli
            holdconst (list of boolean): which of the parameters to hold constant (in a sequence of [*lps, *lcs, *Ks])
        '''
        force = forceExtension.values
        extension = np.array(forceExtension.coords['extension'])
        
        init = [*lps, *lcs, *Ks]
        #decimate the waves to get 1 ms time resolution
        if(len(force) > 5000):
            force_decimated = force[::100]
            extension_decimated = extension[::100]
        else: #not enough points to fit
            warnings.warn("Curve fit failed: not enough points")
            fitfailed = True
            return -1, -1, -1, fitfailed

        bounds_upper = []
        bounds_lower = []
        #create array of bounds to account for parameters we wish to hold constant:
        for const, param in zip(holdconst, init):
            if const:
                bounds_upper.append(1.001 * param)
                bounds_lower.append(0.999 * param)
            else:
                bounds_upper.append(np.inf)
                bounds_lower.append(-np.inf)

        bounds = (bounds_lower, bounds_upper)

        try:
            params, params_cov = curve_fit(self._fit2WLCs, force_decimated, extension_decimated, p0=init, bounds = bounds)
        except RuntimeError:
            fitfailed = True
            return -1, -1, -1, fitfailed

        #generate a fit object and return it
        fitforce = np.arange(1e-12, 70e-12, 1e-12)
        fitextension = self._fit2WLCs(fitforce, *params)
        fitArray = xr.DataArray(fitforce,
                                dims=['extension'],
                                coords={'extension': fitextension},
                                name='force')

        return params, params_cov, fitArray, False


    def _fit2WLCs(self, F, lp_anchor, lp_protein, lc_anchor, lc_protein, K_anchor, K_protein):

        return xSeriesWLCe(F, [lp_anchor, lp_protein], [lc_anchor, lc_protein], [K_anchor, K_protein])





    