#!/usr/bin/env python3

import sys
import time
import copy
import subprocess
import numpy as np
import mahotas as mh
import cv2 as cv
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
#from matplotlib import pyplot as plt
from scipy import ndimage as ndi # ndi.fourier_shift
from scipy.signal import windows, correlate2d
from scipy.interpolate import griddata
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import (
					FigureCanvasQTAgg as FigureCanvas,
					NavigationToolbar2QT as NavigationToolbar
							)
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib import colors, ticker, cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtCore import (
					Qt, QPoint, QRect, QSize,
					QObject, QThread, pyqtSignal,
					QEvent
							)
from PyQt5.QtGui import (
					QIntValidator, QDoubleValidator,
					QMouseEvent, QPalette, QColor
							)
from PyQt5.QtWidgets import (
					QApplication, QLabel, QWidget, QFrame,
					QPushButton, QHBoxLayout, QVBoxLayout,
					QComboBox, QCheckBox, QSlider, QProgressBar,
					QFormLayout, QLineEdit, QTabWidget,
					QSizePolicy, QFileDialog, QMessageBox,
					QInputDialog, QWidget, QListWidget,
					QGroupBox, QMenu
							)
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.util import view_as_windows
from pathlib import Path
from readPTU_FLIM import PTUreader
from scipy import optimize
from scipy.ndimage import gaussian_filter
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp

################################################################################
# colormaps for matplotlib #
############################

red_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
red_cmap = LinearSegmentedColormap('red_cmap', red_cdict)
#cm.register_cmap(cmap=red_cmap)
try:
	matplotlib.colormaps.register(red_cmap)
except:
	pass

green_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
green_cmap = LinearSegmentedColormap('green_cmap', green_cdict)
#cm.register_cmap(cmap=green_cmap)
try:
	matplotlib.colormaps.register(green_cmap)
except:
	pass

blue_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
blue_cmap = LinearSegmentedColormap('blue_cmap', blue_cdict)
#cm.register_cmap(cmap=blue_cmap)
try:
	matplotlib.colormaps.register(blue_cmap)
except:
	pass

transparent_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'alpha': ((0, 1.0, 1.0),
				  (1, 0.0, 0.0)),
			}
transparent_cmap = LinearSegmentedColormap('transparent_cmap',
											transparent_cdict)
#cm.register_cmap(cmap=transparent_cmap)
try:
	matplotlib.colormaps.register(transparent_cmap)
except:
	pass

################################################################################
# remove tree method for pathlib #
##################################

def rm_tree(pth):
	pth = Path(pth)
	for child in pth.glob('*'):
		if child.is_file():
			child.unlink()
		else:
			rm_tree(child)
	pth.rmdir()

################################################################################
# method for windowing matrix
################################################################################

def window_over(arr, size = 2, axes = (0,1) ):
	wshp = list(arr.shape)
	for a in axes:
		wshp[a] = size
	return view_as_windows(arr, wshp, wshp).squeeze()

################################################################################
# helper functions for GUI elements #
#####################################

def display_error (error_text = 'Something went wrong!'):
	msg = QMessageBox()
	msg.setIcon(QMessageBox.Critical)
	msg.setText("Error")
	msg.setInformativeText(error_text)
	msg.setWindowTitle("Error")
	msg.exec_()

def setup_textbox (function, layout, label_text,
				   initial_value = 0):
	textbox = QLineEdit()
	need_inner = not isinstance(layout, QHBoxLayout)
	if need_inner:
		inner_layout = QHBoxLayout()
	label = QLabel(label_text)
	label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
	if need_inner:
		inner_layout.addWidget(label)
	else:
		layout.addWidget(label)
	textbox.setMaxLength(6)
	textbox.setFixedWidth(50)
	textbox.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
	textbox.setValidator(QDoubleValidator())
	textbox.setText(str(initial_value))
	textbox.editingFinished.connect(function)
	if need_inner:
		inner_layout.addWidget(textbox)
		layout.addLayout(inner_layout)
	else:
		layout.addWidget(textbox)
	return textbox

def get_textbox (textbox,
				 minimum_value = None,
				 maximum_value = None,
				 is_int = False):
	if is_int:
		value = int(np.floor(float(textbox.text())))
	else:
		value = float(textbox.text())
	if maximum_value is not None:
		if value > maximum_value:
			value = maximum_value
	if minimum_value is not None:
		if value < minimum_value:
			value = minimum_value
	textbox.setText(str(value))
	return value

def setup_button (function, layout, label_text, toggle = False):
	button = QPushButton()
	if toggle:
		button.setCheckable(True)
	button.setText(label_text)
	button.clicked.connect(function)
	layout.addWidget(button)
	return button

def setup_checkbox (function, layout, label_text,
					is_checked = False):
	checkbox = QCheckBox()
	checkbox.setText(label_text)
	checkbox.setChecked(is_checked)
	checkbox.stateChanged.connect(function)
	layout.addWidget(checkbox)
	return checkbox

def setup_list (function, layout, label_text):
	list_widget = QListWidget()
	list_widget.clicked.connect(function)
	layout.addWidget(list_widget)
	return list_widget

def setup_tab (tabs, tab_layout, label):
	tab = QWidget()
	tab.layout = QVBoxLayout()
	tab.setLayout(tab.layout)
	tab.layout.addLayout(tab_layout)
	tabs.addTab(tab, label)

def horizontal_separator (layout, palette):
	separator = QFrame()
	separator.setFrameShape(QFrame.HLine)
	#separator.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Expanding)
	separator.setLineWidth(1)
	palette.setColor(QPalette.WindowText, QColor('lightgrey'))
	separator.setPalette(palette)
	layout.addWidget(separator)

def setup_progress_bar (layout):
	progress_bar = QProgressBar()
	clear_progress_bar(progress_bar)
	layout.addWidget(progress_bar)
	return progress_bar

def clear_progress_bar (progress_bar):
	progress_bar.setMinimum(0)
	progress_bar.setFormat('')
	progress_bar.setMaximum(1)
	progress_bar.setValue(0)

def update_progress_bar (progress_bar, value = None,
						 minimum_value = None,
						 maximum_value = None,
						 text = None):
	if minimum_value is not None:
		progress_bar.setMinimum(minimum_value)
	if maximum_value is not None:
		progress_bar.setMaximum(maximum_value)
	if value is not None:
		progress_bar.setValue(value)
	if text is not None:
		progress_bar.setFormat(text)

def setup_slider (function, layout, minimum_value = 0, maximum_value = 1,
				  start_value = 0, step_size = 1, direction = Qt.Horizontal):
		slider = QSlider(direction)
		slider.setMinimum(0)
		slider.setMaximum(maximum_value)
		slider.setSingleStep(step_size)
		slider.setValue(start_value)
		slider.valueChanged.connect(function)
		layout.addWidget(slider)
		return slider

def update_slider (slider, value = None,
				   maximum_value = None):
	if value is not None:
		slider.setValue(value)
	if maximum_value is not None:
		slider.setMaximum(maximum_value)

def setup_combobox (function, layout, label_text):
	combobox = QComboBox()
	need_inner = not isinstance(layout, QHBoxLayout)
	if need_inner:
		inner_layout = QHBoxLayout()
	label = QLabel(label_text)
	label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
	if need_inner:
		inner_layout.addWidget(label)
	else:
		layout.addWidget(label)
	combobox.currentIndexChanged.connect(function)
	if need_inner:
		inner_layout.addWidget(combobox)
		layout.addLayout(inner_layout)
	else:
		layout.addWidget(combobox)
	return combobox

def setup_labelbox (label_text, initial_text):
	text_box = QFrame()
	layout = QHBoxLayout()
	text_box.setFrameShape(QFrame.StyledPanel)
#	self.instruction_box.setSizePolicy(QSizePolicy.Expanding)
	label = QLabel(label_text)
	label.setAlignment(Qt.AlignLeft)
	text = QLabel(initial_text)
	text.setAlignment(Qt.AlignLeft)
#	self.instruction_text.setWordWrap(True)
	layout.addWidget(label)
	layout.addWidget(text)
	layout.addStretch()
	text_box.setLayout(layout)
	return text_box, text

def clear_layout (layout):
	for i in reversed(range(layout.count())): 
		widgetToRemove = layout.takeAt(i).widget()
		layout.removeWidget(widgetToRemove)
		widgetToRemove.deleteLater()

################################################################################
# Functions used for fitting.
################################################################################

# Mono-Exponential model
def ME(x,A,tau):
	return A*np.exp(-x/tau)

# Bi-Exponential model
def BE(x,A,tau1,B,tau2):
	return A*np.exp(-x/tau1)+B*np.exp(-x/tau2)

# Instrument Response Function assuming a gaussian profile
def IRF(x,mu,sigma):
	return np.exp(-(x-mu)**2/2/sigma**2)/sigma/np.sqrt(2*np.pi)

# Mono-Exponential convolution
def MEC(x,A,tau1,mu,sigma):
	return np.fft.ifft(np.fft.fft(IRF(x,mu,sigma)) * \
					np.fft.fft(ME(x,A,tau1))).real * (x[1]-x[0])

# Bi-Exponential convolution
def BEC(x,B,tau2,A,tau1,mu,sigma):
	return np.fft.ifft(np.fft.fft(IRF(x,mu,sigma)) * \
					np.fft.fft(BE(x,A,tau1,B,tau2))).real * (x[1]-x[0])

# Negative Log-Likelihood estimator assuming Poisson statistics
def NLL(p, X, Y, F, startpoint=0, endpoint=-1):
	if endpoint == -1:
		endpoint = len(X)
	FX = F(X, *p)[startpoint:endpoint]
	RY = Y[startpoint:endpoint]
	return np.sum(FX - RY*np.log(FX)) / \
						np.sqrt(endpoint - startpoint)

#TODO: Not sure this is quite right!
def CHI(p, X, Y, F, startpoint=0, endpoint=-1):
	if endpoint == -1:
		endpoint = len(X)
	FX = F(X, *p)[startpoint:endpoint]
	RY = Y[startpoint:endpoint]
	return np.sum(np.divide((FX - RY)**2, FX)) / \
									(endpoint - startpoint)
#	return np.sum((FX - RY)**2) #/ (endpoint - startpoint)

# Minimse the Negative Log-Likelihood for a given function and dataset
def perform_fit(X, Y, F, p, startpoint=0, endpoint=-1, fit_type = 'NLL'):
	if fit_type == 'NLL':
		fit_function = lambda p, X, Y: NLL(p, X, Y, F, startpoint, endpoint)
	else:
		fit_function = lambda p, X, Y: CHI(p, X, Y, F, startpoint, endpoint)
	return optimize.fmin(fit_function, p, args=(X, Y),
						disp=False, full_output=True )

################################################################################
# Weiner Deconvolution on data. Used for Tail Fitting.
################################################################################

def WeinerDeconvolution (time_points, data_points, mu, sigma, alpha = 60.):
	time_zero = time_points(np.argmax(data_points)) - sigma
	H = np.fft.fftshift(np.fft.fft(data_points))
	G = np.fft.fftshift(np.fft.fft(IRF(time_points,time_zero,sigma)))
	M = (np.conj(G)/(np.abs(G)**2 + alpha**2))*H
	m = np.abs(H.shape[0]*np.fft.ifft(M).real)
	return m/np.amax(m)

################################################################################
# Fit for different endpoints.
################################################################################

def find_endpoint(time_points, data_points,
					fit_function, guess_params,
					startpoint = 0,
					fit_type = 'NLL', # 'CHI' for Chi Square
					coarse_N = 10, fine_N = 10):
	peak_index = np.argmax(data_points)
	peak_value = data_points[peak_index]
	filtered_data = gaussian_filter(data_points,10,mode='constant')
	mask = filtered_data[peak_index:] < peak_value * 0.2
	initial_index = peak_index + np.amin(np.where(mask))
	index_range = len(time_points) - initial_index
	total_N = coarse_N + fine_N
	endpoints = np.zeros(total_N,dtype=int)
	likelihoods = np.zeros(total_N)
	fit_params = np.zeros((total_N,len(guess_params)))
	for i in range(coarse_N):
		endpoint = int(initial_index + np.floor(i*index_range/coarse_N))
		endpoints[i] = endpoint
		fit = perform_fit(time_points, data_points,
						 fit_function, guess_params,
						 startpoint = startpoint,
						 endpoint = endpoint,
						 fit_type = fit_type)
		fit_params[i] = fit[0]
		likelihoods[i] = fit[1]
	iMax = np.argmax(likelihoods[:coarse_N])
	fine_upper = min(len(time_points),
				initial_index + np.floor((iMax+1)*index_range/coarse_N))
	fine_lower = max(initial_index,
				initial_index + np.floor((iMax-1)*index_range/coarse_N))
	fine_range = fine_upper - fine_lower
	for i in range(coarse_N, coarse_N + fine_N):
		endpoint = int(fine_lower + np.floor((i-coarse_N)*fine_range/fine_N))
		endpoints[i] = endpoint
		fit = perform_fit(time_points, data_points,
						 fit_function, guess_params,
						 startpoint = startpoint,
						 endpoint = endpoint,
						 fit_type = fit_type)
		fit_params[i] = fit[0]
		likelihoods[i] = fit[1]
	return fit_params, endpoints, likelihoods#/np.amax(likelihoods)

################################################################################
# Cut Data bassed on given thresholds as factor of peak
################################################################################

def cut_data (time_points, data_points,
				lower_threshold = 0.01,
				upper_threshold = 0.02):
	peak_index = np.argmax(data_points)
	peak_value = data_points[peak_index]
	mask = data_points[peak_index:] < peak_value*upper_threshold
	if np.any(mask):
		upper_bound = np.argmax(mask) + peak_index
	else:
		upper_bound = len(data_points)
	mask = data_points[:peak_index] > peak_value*lower_threshold
	if np.any(mask):
		lower_bound = np.argmax(mask)
	else:
		lower_bound = 0
	return time_points[lower_bound:upper_bound],\
			data_points[lower_bound:upper_bound]

################################################################################
#
################################################################################

def fit_data (fit_function, initial_guess, time_points, data_points):
	time_points, data_points = cut_data(time_points, data_points)
	total_photons = np.sum(data_points)
	peak_photons = np.amax(data_points)
	data_points = data_points / peak_photons
	startpoint = np.amax([0, np.argmax(data_points)-24])
	fit_params, endpoints, likelihoods = find_endpoint(
									time_points, data_points,
									fit_function, initial_guess,
									startpoint = startpoint)
	best_fit = np.argmax(likelihoods)
	endpoint = endpoints[best_fit]
	best_params = fit_params[best_fit]
	likelihood = likelihoods[best_fit]
	results = FitResults( fit_function, best_params,
							time_points, data_points,
							startpoint, endpoint,
							total_photons, peak_photons )
	return results

################################################################################
#
################################################################################

def quick_fit_data (fit_function, initial_guess, time_points, data_points):
#	time_points, data_points = cut_data(time_points, data_points)
	total_photons = np.sum(data_points)
	peak_photons = np.amax(data_points)
	data_points = data_points / peak_photons
	startpoint = 0
	endpoint = -1
	fit = perform_fit(time_points, data_points,
					  fit_function, initial_guess)
	best_params = fit[0]
	results = FitResults( fit_function, best_params,
							time_points, data_points,
							startpoint, endpoint,
							total_photons, peak_photons )
	return results

################################################################################
# data structure for fit results
################################################################################

class FitResults ():
	def __init__ (self, fit_function, best_params,
					time_points, data_points,
					startpoint, endpoint,
					total_photons, peak_photons):
		self.fit_function = fit_function
		self.best_params = best_params
		self.time_points = time_points
		self.data_points = data_points
		self.startpoint = startpoint
		self.endpoint = endpoint
		self.total_photons = total_photons
		self.peak_photons = peak_photons

################################################################################
# matplotlib canvas widget #
############################

class MPLCanvas(FigureCanvas):
	def __init__ (self, parent=None, width=10, height=8, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
	#	(self.ax, self.cax) = self.fig.subplots(1, 2, width_ratios=[8,1])
		self.ax = self.fig.subplots(1,1)
		self.divider = make_axes_locatable(self.ax)
		self.cax = self.divider.append_axes('right', size='5%', pad=0.05)
		self.cax.yaxis.tick_right()
		self.cax.set_xticks([])
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		self.fig.set_tight_layout(True)
		# stuff to plot
		self.image_array = np.array([[0]], dtype = int)
		self.heatmap = np.array([[0]], dtype = float)
		self.segments = []
		self.current_segment = None
		# plot objects
		self.image_plot = None
		self.seg_plot = np.array([], dtype = object)
		self.box_plot = None
		self.select_box = None
		self.heatmap_plot = None
		self.segments_plot = None
		self.current_segment_plot = None
		# boolean flags
		self.show_box = False
		self.show_seg = np.array([], dtype = bool)
		# colour choices
		self.colormap = 'Greys_r'
		self.available_colormaps = ['jet', 'plasma_r',
									'plasma', 'viridis',
									'afmhot']
		self.heat_colormap = 'jet'
		self.heat_alpha = 0.3
		self.seg_color = 'white'
		self.seg_alpha = 0.5
		self.show_heatmap = True
		#
		self.zoomed = False
		self.flip_vertical = False
		self.plot_image()
		#
		self.vmax = 3.7
		self.vmin = 2.5
		self.gaussian_factor = 1
	
	def clear_canvas (self):
		# stuff to plot
		self.image_array = np.array([[0]], dtype = int)
		self.heatmap = np.array([[0]], dtype = float)
		# plot objects
		self.image_plot = None
		self.seg_plot = np.array([], dtype = object)
		self.box_plot = None
		self.select_box = None
		self.heatmap_plot = None
		# boolean flags
		self.show_box = False
		self.show_seg = np.array([], dtype = bool)
		# clear images
		self.ax.clear()
		self.draw()
	
	def set_flip (self, flip_vertical = False):
		self.flip_vertical = flip_vertical
		self.set_bounds()
		self.draw()
	
	def set_zoom (self, zoomed = False):
		self.zoomed = zoomed
		self.set_bounds()
		self.draw()
	
	def set_bounds (self):
		if self.zoomed and self.focus_box is not None:
			self.ax.set_xlim(
						left = self.focus_box[0,0],
						right = self.focus_box[0,1])
			self.ax.set_ylim(
						bottom = self.focus_box[1,0],
						top = self.focus_box[1,1])
		else:
			self.ax.set_xlim(left = 0, right = self.image_array.shape[1])
			self.ax.set_ylim(bottom = 0, top = self.image_array.shape[0])
		if self.flip_vertical:
			self.ax.invert_yaxis()
	
	def update_image (self, image_array = np.array([[0]], dtype = int)):
		self.image_array = image_array
		self.plot_image()
	
	def update_heatmap (self, heatmap = np.array([[0]], dtype = float)):
		self.heatmap = heatmap
		self.plot_heatmap()
	
	def update_focus_box (self, focus_box):
		self.focus_box = focus_box
		self.plot_box()
	
	def update_colors (self, colormap = 'Greys_r', #'afmhot'
							 heat_colormap = 'plasma_r',
							 seg_color = 'white', # 'blue'
							 seg_alpha = 0.5):
		self.colormap = colormap
		self.heat_colormap = heat_colormap
		self.seg_color = marker_color
		self.seg_alpha = marker_alpha
	
	def update_segments (self, segments = [], current_segment = None):
		self.segments = segments
		self.current_segment = current_segment
		self.plot_segments()
	
	def plot_box (self):
		self.remove_plot_element(self.box_plot)
		focus_box = self.focus_box.copy()
		if self.show_box:
			self.box_plot = self.ax.plot((focus_box[0,0],
										  focus_box[0,1],
										  focus_box[0,1],
										  focus_box[0,0],
										  focus_box[0,0]),
										 (focus_box[1,0],
										  focus_box[1,0],
										  focus_box[1,1],
										  focus_box[1,1],
										  focus_box[1,0]),
										 color='gold',
										 linestyle='-',
										 linewidth = 1,
										 alpha = 0.8,
										 zorder = 6)
		else:
			self.box_plot = None
		self.draw()
	
	def plot_selector (self, p_1, p_2):
		self.remove_selector()
		self.select_box = self.ax.plot((p_1[0], p_2[0], p_2[0], p_1[0],
										p_1[0]),
									   (p_1[1], p_1[1], p_2[1], p_2[1],
										p_1[1]),
									color = 'white',
									linestyle = '-',
									linewidth = 1,
									zorder = 7)
		self.draw()
	
	def remove_selector (self):
		if self.select_box is not None:
			if isinstance(self.select_box,list):
				for line in self.select_box:
					line.remove()
			else:
				self.select_box.remove()
			self.select_box = None
		self.draw()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for line in plot_element:
					try:
						line.remove()
					except:
						pass
			else:
				try:
					plot_element.remove()
				except:
					pass
	
	def plot_image (self):
		self.remove_plot_element(self.image_plot)
		self.image_plot = self.ax.imshow(self.image_array,
										 cmap = self.colormap,
										 zorder = 1)
		self.set_bounds()
		self.draw()
	
	def plot_heatmap (self):
		self.remove_plot_element(self.heatmap_plot)
		if self.heatmap is not None:
			heat_alpha = np.zeros_like(self.heatmap)
			if self.show_heatmap:
				heat_alpha[self.heatmap > 0] = self.heat_alpha
			self.heatmap = np.ma.array(self.heatmap, mask = self.heatmap == 0)
			heatmap = self.heatmap
			if self.gaussian_factor > 1:
				heatmap = gaussian_filter(self.heatmap,
											sigma = self.gaussian_factor)
		#	self.heat_colormap.set_bad()
			self.heatmap_plot = self.ax.imshow(heatmap,
											   cmap = self.heat_colormap,
											   alpha = heat_alpha,
											   vmax = self.vmax,
											   vmin = self.vmin,
											   zorder = 2)
			self.fig.colorbar(self.heatmap_plot, cax=self.cax,
								orientation='vertical')
		self.draw()
	
	def plot_segments (self):
		self.remove_plot_element(self.segments_plot)
		self.remove_plot_element(self.current_segment_plot)
		if self.segments is not None:
			segments_array = np.zeros_like(self.image_array)
			for segment in self.segments:
				segments_array[segment] = 1
			self.segments_plot = self.ax.imshow(segments_array,
												cmap = self.colormap,
												alpha = 0.2*segments_array,
												zorder = 3)
			if self.current_segment is not None:
				if len(self.segments) > 0:
					current_segment_array = self.segments[self.current_segment]
					self.current_segment_plot = self.ax.imshow(
											current_segment_array,
											cmap = self.colormap,
											alpha = 0.2*current_segment_array,
											zorder = 4)
		self.draw()

################################################################################
# mpl canvas for simple plots #
###############################

class MPLPlot(FigureCanvas):
	def __init__ (self, parent=None, width=8, height=4, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.add_subplot(111)
		self.ax.set_xlim([0,1])
		self.ax.set_ylim([0.01,1])
		self.ax.set_yscale('log')
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		self.fig.set_tight_layout(True)
		# stuff to plot
		self.results = None
		# plot objects
		self.point_plot = None
		self.line_plot = None
		self.dotted_plot = None
		self.legend = None
	
	def update_plot (self, results = None):
		if results is not None:
			self.results = results
		self.plot()
	
	def clear_canvas (self):
		# stuff to plot
	#	self.points = np.array([[]], dtype = float)
	#	self.chosen = np.array([], dtype = bool)
	#	self.params = np.array([], dtype = float)
		# plot objects
		self.remove_plot_element(self.point_plot)
		self.point_plot = None
		self.remove_plot_element(self.line_plot)
		self.line_plot = None
		self.remove_plot_element(self.dotted_plot)
		self.dotted_plot = None
		self.remove_plot_element(self.legend)
		self.legend = None
		self.ax.cla()
		#
		self.draw()
	
	def plot (self):
		self.clear_canvas()
		fit_function = self.results.fit_function
		best_params = self.results.best_params
		time_points = self.results.time_points
		data_points = self.results.data_points
		startpoint = self.results.startpoint
		endpoint = self.results.endpoint
		fit_time_points = np.linspace(np.amin(time_points),
									  np.amax(time_points), 1000)
		fit_data_points = fit_function(fit_time_points, *best_params)
	#	fit_points = fit_function(time_points, *best_params)
		peak_index = np.argmax(data_points)
		fit_peak_index = np.argmax(fit_data_points)
		self.ax.plot(time_points[startpoint:endpoint],
				data_points[startpoint:endpoint],
				marker = '.',
				linestyle = 'none',
				color = 'tab:blue',
				label = 'Data')
		self.ax.plot(fit_time_points, fit_data_points,
				#	time_points[startpoint:endpoint],
				#	fit_points[startpoint:endpoint],
						linestyle = 'solid',
						color = 'tab:red',
						label = 'Full Fit')
		a = fit_data_points[-1] / \
					np.exp(-fit_time_points[-1]/best_params[1])
		self.ax.plot(fit_time_points[fit_peak_index:],
				a*np.exp(-fit_time_points[fit_peak_index:]/best_params[1]),
			#	time_points[peak_index:endpoint],
			#	a*np.exp(-time_points[peak_index:endpoint]/best_params[1]),
				linestyle = 'dashed',
				color = 'tab:orange',
				label = r'Signal Fit ($\tau = ' + \
						'{0:.3f}ns'.format(best_params[1]) + r'$)')
		self.ax.set_yscale('log')
		self.ax.set_xlabel('Time (ns)')
		self.ax.set_ylim([data_points[endpoint] * 0.9,
					data_points[peak_index] * 1.1])
		self.ax.set_xlim([time_points[peak_index] - 0.3,
					time_points[endpoint] + 0.1])
		self.ax.legend()
		self.draw()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for line in plot_element:
					try:
						line.remove()
					except:
						pass
			else:
				plot_element.remove()



################################################################################
# main window object #
######################

class Window(QWidget):
	def __init__ (self):
		super().__init__()
		self.title = "FLIM Fitting Tool"
		self.canvas = MPLCanvas()
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.plot_canvas = MPLPlot()
		self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
		self.selecting_area = False
		self.click_id = 0
		self.move_id = 0
		self.position = np.array([0,0])
		#
		self.file_path = None
		#
		self.data_stack = np.array([], dtype = int)
		self.image_array = np.array([[0]], dtype = int)
		self.header = None
		self.segments = []
		self.segs_renamed = []
		self.grid_heatmap = None
		self.segment_heatmap = None
		#
		self.x_size = 1
		self.x_lower = 0
		self.x_upper = self.x_size
		self.y_size = 1
		self.y_lower = 0
		self.y_upper = self.y_size
		self.zoomed = False
		self.channel = 0
		self.use_channel = True
		self.num_channels = 1
		self.xy_res = 1
		self.t_res = 1
		self.grid_factor = 32
		self.use_grid = False
		self.show_segments = True
		self.edit_segments = False
		self.erase_segment = False
		self.seg_paint_mode = 'Select'
		self.brush_size = 5
		self.photon_threshold = 8000
		self.use_af = True
		self.fit_each_af = False
		self.peak_index = 0
		self.af_lifetime = 0.4
		self.irf_centre = -0.05
		self.irf_width = 0.05
		self.irf_centre_guess = -0.12
		self.irf_width_guess = 0.08
		self.af_fraction_guess = 0.3
		self.af_life_guess = 0.4
		self.lifetime_guess = 3.
		self.grid_results = None
		self.segment_results = None
		self.full_field_results = None
		self.lifetime_max = 3.7
		self.lifetime_min = 2.5
		self.colour_max = self.lifetime_max
		self.colour_min = self.lifetime_min
		self.colour_alpha = 0.3
		self.show_heatmap = True
		self.gaussian_factor = 1
		self.startpoint = 0
		self.endpoint = -1
		#
		self.progress_counter = 0
		self.cores_to_use = mp.cpu_count() - 1
		#
		self.setupGUI()
		#
		self.click_id = self.canvas.mpl_connect(
							'button_press_event', self.on_click)
	
	def setupGUI (self):
		self.setWindowTitle(self.title)
		# layout for full window
		outer_layout = QVBoxLayout()
		# top section for plot and options
		main_layout = QHBoxLayout()
		# main left for plot
		plot_layout = QVBoxLayout()
		plot_layout.addWidget(self.canvas)
		self.canvas.resize(self.canvas.sizeHint())
		plot_layout.addWidget(self.toolbar)
		plot_layout.addWidget(self.plot_canvas)
		plot_layout.addWidget(self.plot_toolbar)
		main_layout.addLayout(plot_layout)
		# main right for options
		options_layout = QHBoxLayout()
		# options tabs
		tabs = QTabWidget()
		tabs.setMinimumWidth(220)
		tabs.setMaximumWidth(220)
		focus_layout = self.setup_focus_layout()
		setup_tab(tabs, focus_layout, 'focus')
	#	segment_layout = self.setup_segment_layout()
	#	setup_tab(tabs, segment_layout, 'segments')
		fit_layout = self.setup_fit_layout()
		setup_tab(tabs, fit_layout, 'fitting')
		options_layout.addWidget(tabs)
		#
		main_layout.addLayout(options_layout)
		outer_layout.addLayout(main_layout)
		result_box, self.result_text = setup_labelbox(
						'<font color="red">Fit Info: </font>',
						'Fitting not done.')
		outer_layout.addWidget(result_box)
		# horizontal row of buttons
		outer_layout.addLayout(self.setup_bottom_layout())
		# instructions box
		instruction_box, self.instruction_text = setup_labelbox(
						'<font color="red">File Info: </font>',
						'"Open File" to begin.')
		outer_layout.addWidget(instruction_box)
		# Set the window's main layout
		self.setLayout(outer_layout)
	
	def setup_focus_layout (self):
		focus_layout = QVBoxLayout()
		#
		self.checkbox_channel = QGroupBox('Use Single Channel')
		self.checkbox_channel.setCheckable(True)
		self.checkbox_channel.setChecked(self.use_channel)
		self.checkbox_channel.toggled.connect(self.refresh_image)
		channel_layout = QVBoxLayout()
		self.channel_box = setup_combobox(
							self.select_channel,
							channel_layout, 'Channel:')
		self.checkbox_channel.setLayout(channel_layout)
		focus_layout.addWidget(self.checkbox_channel)
	#	focus_layout.addWidget(QLabel('XY Working Space'))
		xy_box = QGroupBox('XY Working Space')
		xy_layout = QVBoxLayout()
		self.button_select = setup_button(
							self.select_bounds,
							xy_layout, 'Select Box')
		self.button_reset = setup_button(
							self.reset_bounds,
							xy_layout, 'Select All')
		self.textbox_x_min = setup_textbox(
							self.bound_textbox_select,
							xy_layout, 'X Min:')
		self.textbox_x_max = setup_textbox(
							self.bound_textbox_select,
							xy_layout, 'X Max:')
		self.textbox_y_min = setup_textbox(
							self.bound_textbox_select,
							xy_layout, 'Y Min:')
		self.textbox_y_max = setup_textbox(
							self.bound_textbox_select,
							xy_layout, 'Y Max:')
		self.checkbox_zoom = setup_checkbox(
							self.zoom_checkbox,
							xy_layout, 'zoomed',
							self.zoomed)
		self.checkbox_flip = setup_checkbox(
							self.flip_checkbox,
							xy_layout, 'flipped',
							False)
		#TODO: Crop Image Button
		xy_box.setLayout(xy_layout)
		focus_layout.addWidget(xy_box)
		#
	#	focus_layout.addWidget(QLabel('Segmentation'))
		seg_box = QGroupBox('Segmentation')
		seg_layout = QVBoxLayout()
		grid_layout = QHBoxLayout()
		self.checkbox_grid = setup_checkbox(
							self.grid_checkbox,
							grid_layout, 'Grid',
							self.use_grid)
		self.textbox_grid = setup_textbox(
							self.bound_textbox_select,
							grid_layout, 'Box Size:',
							self.grid_factor)
		seg_layout.addLayout(grid_layout)
		self.seg_list = setup_list(
							self.select_segment,
							seg_layout, 'Segments')
		self.seg_list.installEventFilter(self)
		seg_buttons_layout = QHBoxLayout()
		self.button_add_segment = setup_button(
							self.add_segment,
							seg_buttons_layout, 'Add')
		self.button_add_segment = setup_button(
							self.remove_segment,
							seg_buttons_layout, 'Remove')
		seg_layout.addLayout(seg_buttons_layout)
		self.checkbox_show_segs = setup_checkbox(
							self.show_segs_checkbox,
							seg_layout, 'show segments',
							self.show_segments)
		paint_buttons_layout = QHBoxLayout()
		self.button_seg_select = setup_button(
							self.seg_select_select,
							paint_buttons_layout, 'Select',
							toggle = True)
		self.button_seg_select.setChecked(True)
		self.button_seg_paint = setup_button(
							self.seg_paint_select,
							paint_buttons_layout, 'Paint',
							toggle = True)
		self.button_seg_erase = setup_button(
							self.seg_erase_select,
							paint_buttons_layout, 'Erase',
							toggle = True)
		seg_layout.addLayout(paint_buttons_layout)
		brush_box = QGroupBox('brush size')
		brush_layout = QHBoxLayout()
		self.slider_brush_size = setup_slider(
							self.select_brush_size,
							brush_layout,
							minimum_value = 0,
							maximum_value = 12,
							start_value = self.brush_size)
		brush_box.setLayout(brush_layout)
		seg_layout.addWidget(brush_box)
		self.button_clear_segments = setup_button(
							self.clear_segments,
							seg_layout, 'Clear Segments')
		self.button_import_segments = setup_button(
							self.import_segments,
							seg_layout, 'Import Segments')
		self.button_export_segments = setup_button(
							self.export_segments,
							seg_layout, 'Export Segments')
		self.button_flip_segments = setup_button(
							self.flip_segments,
							seg_layout, 'Flip Segments')
		
		seg_box.setLayout(seg_layout)
		focus_layout.addWidget(seg_box)
		#
	#	focus_layout.addStretch()
		#
		self.setup_bound_textboxes()
		return focus_layout
	
	#def setup_segment_layout (self):
	#	segment_layout = QVBoxLayout()
	#	return segment_layout
	
	def setup_fit_layout (self):
		fit_layout = QVBoxLayout()
		self.textbox_photon = setup_textbox(
							self.fit_textbox_select,
							fit_layout, 'Photon Thresh:',
							self.photon_threshold)
		self.textbox_min_lifetime = setup_textbox(
							self.fit_textbox_select,
							fit_layout, 'Min Lifetime:',
							self.lifetime_min)
		self.textbox_max_lifetime = setup_textbox(
							self.fit_textbox_select,
							fit_layout, 'Max Lifetime:',
							self.lifetime_max)
		self.textbox_irf_centre = setup_textbox(
							self.fit_textbox_select,
							fit_layout, 'IRF Centre:',
							self.irf_centre)
		self.textbox_irf_width = setup_textbox(
							self.fit_textbox_select,
							fit_layout, 'IRF Width:',
							self.irf_width)
#		self.checkbox_use_af = setup_checkbox(
#							self.use_af_checkbox,
#							fit_layout, 'autofluorescence',
#							self.use_af)
		self.checkbox_use_af = QGroupBox('autofluorescence')
		self.checkbox_use_af.setCheckable(True)
		self.checkbox_use_af.setChecked(self.use_af)
		self.checkbox_use_af.toggled.connect(self.fit_textbox_select)
	#	self.checkbox_use_af.stateChanged.connect(self.use_af_checkbox)
		af_box_layout = QVBoxLayout()
		self.textbox_af_lifetime = setup_textbox(
							self.fit_textbox_select,
							af_box_layout, 'AF Time Const:',
							self.af_lifetime)
		self.checkbox_use_af.setLayout(af_box_layout)
		fit_layout.addWidget(self.checkbox_use_af)
		self.textbox_irf_cen_guess = setup_textbox(
							self.fit_textbox_select,
							fit_layout, 'IRF Centre Guess:',
							self.irf_centre_guess)
		self.textbox_irf_wid_guess = setup_textbox(
							self.fit_textbox_select,
							fit_layout, 'IRF Width Guess:',
							self.irf_width_guess)
		self.button_fit_irf = setup_button(
							self.fit_irf,
							fit_layout, 'Fit IRF (and AF)')
		self.checkbox_fit_each = QGroupBox('fit each AF')
		self.checkbox_fit_each.setCheckable(True)
		self.checkbox_fit_each.setChecked(self.fit_each_af)
		self.checkbox_fit_each.toggled.connect(self.fit_textbox_select)
		each_box_layout = QVBoxLayout()
		self.textbox_af_frac_guess = setup_textbox(
							self.fit_textbox_select,
							each_box_layout, 'Fraction Guess:',
							self.af_fraction_guess)
		self.textbox_af_life_guess = setup_textbox(
							self.fit_textbox_select,
							each_box_layout, 'AF Life Guess:',
							self.af_life_guess)
		self.checkbox_fit_each.setLayout(each_box_layout)
		fit_layout.addWidget(self.checkbox_fit_each)
		self.textbox_lifetime_guess = setup_textbox(
							self.fit_textbox_select,
							fit_layout, 'Lifetime Guess:',
							self.lifetime_guess)
		self.button_fit_all = setup_button(
							self.fit_all,
							fit_layout, 'Fit All Segments')
		self.colormap_box = setup_combobox(self.select_colormap,
							fit_layout, 'Colormap:')
		for colormap in self.canvas.available_colormaps:
			self.colormap_box.addItem(colormap)
		self.colormap_box.setCurrentIndex(0)
		self.textbox_colour_alpha = setup_textbox(
							self.colour_textbox_select,
							fit_layout, 'Opacity:',
							self.colour_alpha)
		self.textbox_colour_min = setup_textbox(
							self.colour_textbox_select,
							fit_layout, 'Min:',
							self.lifetime_min)
		self.textbox_colour_max = setup_textbox(
							self.colour_textbox_select,
							fit_layout, 'Max:',
							self.lifetime_max)
		self.button_reset_colours = setup_button(
							self.reset_colours,
							fit_layout, 'Reset Colour Range')
		self.textbox_gaussian = setup_textbox(
							self.gaussian_textbox_select,
							fit_layout, 'Smoothing:',
							self.gaussian_factor)
		self.checkbox_heatmap = setup_checkbox(
							self.heatmap_checkbox,
							fit_layout, 'show heatmap',
							self.show_heatmap)
		#
		fit_layout.addStretch()
		#
		self.setup_fit_textboxes()
		self.setup_colour_textboxes()
		return fit_layout
	
	def setup_bottom_layout (self):
		bottom_layout = QHBoxLayout()
		self.button_open_file = setup_button(
					self.open_file,
					bottom_layout, 'Open File')
		self.progress_bar = setup_progress_bar(bottom_layout)
		self.button_save_csv = setup_button(
					self.save_csv,
					bottom_layout, 'Save CSV')
		self.button_save_frames = setup_button(
					self.save_image,
					bottom_layout, 'Save Image')
		return bottom_layout
	
	def select_channel (self):
		self.channel = self.channel_box.currentIndex()
		self.refresh_image()
	
	def select_colormap (self):
		self.canvas.heat_colormap = self.colormap_box.currentText()
		self.refresh_heatmap()
	
	def setup_bound_textboxes (self):
		self.textbox_x_min.setText(str(self.x_lower))
		self.textbox_x_max.setText(str(self.x_upper))
		self.textbox_y_min.setText(str(self.y_lower))
		self.textbox_y_max.setText(str(self.y_upper))
	
	def bound_textbox_select (self):
		self.x_lower = get_textbox(self.textbox_x_min,
									minimum_value = 0,
									maximum_value = self.x_size,
									is_int = True)
		self.x_upper = get_textbox(self.textbox_x_max,
									minimum_value = self.x_lower,
									maximum_value = self.x_size,
									is_int = True)
		self.y_lower = get_textbox(self.textbox_y_min,
									minimum_value = 0,
									maximum_value = self.y_size,
									is_int = True)
		self.y_upper = get_textbox(self.textbox_y_max,
									minimum_value = self.y_lower,
									maximum_value = self.y_size,
									is_int = True)
		self.canvas.focus_box = np.array(
									[[self.x_lower, self.x_upper],
									 [self.y_lower, self.y_upper]],
								dtype = int)
		if self.x_lower > 0 or self.x_upper < self.x_size or \
		   self.y_lower > 0 or self.y_upper < self.y_size:
			self.canvas.show_box = True
		else:
			self.canvas.show_box = False
		self.grid_factor = get_textbox(self.textbox_grid,
									minimum_value = 1,
									maximum_value = 128,
									is_int = True)
		self.canvas.plot_box()
	
	def setup_fit_textboxes (self):
		self.textbox_photon.setText(str(self.photon_threshold))
		self.textbox_min_lifetime.setText(str(self.lifetime_min))
		self.textbox_max_lifetime.setText(str(self.lifetime_max))
		self.textbox_irf_centre.setText(str(self.irf_centre))
		self.textbox_irf_width.setText(str(self.irf_width))
		self.textbox_af_lifetime.setText(str(self.af_lifetime))
		self.textbox_irf_cen_guess.setText(str(self.irf_centre_guess))
		self.textbox_irf_wid_guess.setText(str(self.irf_width_guess))
		self.textbox_af_frac_guess.setText(str(self.af_fraction_guess))
		self.textbox_af_life_guess.setText(str(self.af_life_guess))
		self.textbox_lifetime_guess.setText(str(self.lifetime_guess))
	
	def fit_textbox_select (self):
		self.fit_textbox_select = self.checkbox_use_af.isChecked()
		self.fit_each_af = self.checkbox_fit_each.isChecked()
		self.photon_threshold = get_textbox(self.textbox_photon,
											minimum_value = 100,
											is_int = True)
		self.lifetime_min = get_textbox(self.textbox_min_lifetime)
		self.lifetime_max = get_textbox(self.textbox_max_lifetime)
		self.irf_centre = get_textbox(self.textbox_irf_centre,
											minimum_value = -1.,
											maximum_value = 1.)
		self.irf_width = get_textbox(self.textbox_irf_width,
											minimum_value = 0.001,
											maximum_value = 1.)
		self.af_lifetime = get_textbox(self.textbox_af_lifetime)
		self.irf_centre_guess = get_textbox(self.textbox_irf_cen_guess,
											minimum_value = -1.,
											maximum_value = 1.)
		self.irf_width_guess = get_textbox(self.textbox_irf_wid_guess,
											minimum_value = 0.001,
											maximum_value = 1.)
		self.af_fraction_guess = get_textbox(self.textbox_af_frac_guess,
											minimum_value = 0.01,
											maximum_value = 0.99)
		self.af_life_guess = get_textbox(self.textbox_af_life_guess)
		self.lifetime_guess = get_textbox(self.textbox_lifetime_guess)
	
	def colour_textbox_select (self):
		self.colour_min = get_textbox(self.textbox_colour_min)
		self.colour_max = get_textbox(self.textbox_colour_max)
		self.colour_alpha = get_textbox(self.textbox_colour_alpha)
		self.setup_colour_textboxes()
	
	def setup_colour_textboxes (self):
		self.textbox_colour_min.setText(str(self.colour_min))
		self.textbox_colour_max.setText(str(self.colour_max))
		self.textbox_colour_alpha.setText(str(self.colour_alpha))
		self.canvas.vmin = self.colour_min
		self.canvas.vmax = self.colour_max
		self.canvas.heat_alpha = self.colour_alpha
		self.canvas.plot_heatmap()
	
	def gaussian_textbox_select (self):
		self.gaussian_factor = get_textbox(self.textbox_gaussian,
											minimum_value = 1,
											maximum_value = 24,
											is_int = True)
		self.canvas.gaussian_factor = self.gaussian_factor
		self.canvas.plot_heatmap()
	
	def reset_colours (self):
		if self.use_grid:
			if self.grid_heatmap is None:
				return
			self.colour_min = np.amin(self.grid_heatmap[self.grid_heatmap!=0])
			self.colour_max = np.amax(self.grid_heatmap)
		else:
			if self.segment_heatmap is None:
				return
			self.colour_min = np.amin(
								self.segment_heatmap[self.segment_heatmap!=0])
			self.colour_max = np.amax(self.segment_heatmap)
	#	self.colour_alpha = 0.3
		self.setup_colour_textboxes()
	
	def zoom_checkbox (self):
		self.zoomed = self.checkbox_zoom.isChecked()
		self.canvas.set_zoom(self.zoomed)
	
	def flip_checkbox (self):
		flipped = self.checkbox_flip.isChecked()
		self.canvas.set_flip(flipped)
	
	def grid_checkbox (self):
		self.use_grid = self.checkbox_grid.isChecked()
		self.refresh_heatmap()
	
	def show_segs_checkbox (self):
		self.show_segments = self.checkbox_show_segs.isChecked()
		self.canvas.show_segments = self.show_segments
		self.refresh_segments()
	
	def heatmap_checkbox (self):
		self.show_heatmap = self.checkbox_heatmap.isChecked()
		self.canvas.show_heatmap = self.show_heatmap
		self.refresh_heatmap()
	
	def seg_select_select (self):
		self.button_seg_paint.setChecked(False)
		self.button_seg_erase.setChecked(False)
	#	self.seg_paint_mode = 'Select'
		self.edit_segments = False
		self.erase_segment = False
	
	def seg_paint_select (self):
		self.button_seg_select.setChecked(False)
		self.button_seg_erase.setChecked(False)
	#	self.seg_paint_mode = 'Paint'
		self.edit_segments = True
		self.erase_segment = False
	
	def seg_erase_select (self):
		self.button_seg_select.setChecked(False)
		self.button_seg_paint.setChecked(False)
	#	self.seg_paint_mode = 'Erase'
		self.edit_segments = True
		self.erase_segment = True
	
	
	def use_af_checkbox (self):
		self.use_af = self.checkbox_use_af.isChecked()
	
	def add_segment (self):
		self.segments.append(np.zeros((self.y_size,self.x_size), dtype=bool))
		self.segs_renamed.append(False)
		self.seg_list.addItem(f'Segment {len(self.segments):d}')
		self.seg_list.setCurrentRow(len(self.segments)-1)
		self.select_segment()
	
	def remove_segment (self):
		if self.seg_list.selectedItems() is None:
			return
		if len(self.seg_list.selectedItems()) > 0:
			seg_index = self.seg_list.currentRow()
			self.segments.pop(seg_index)
			self.segs_renamed.pop(seg_index)
			self.seg_list.takeItem(seg_index)
			for index in range(seg_index, self.seg_list.count()):
				if not self.segs_renamed[index]:
					self.seg_list.item(index).setText(f'Segment {index+1:d}')
		if len(self.segments) == 0:
			self.seg_list.setCurrentRow(-1)
		self.canvas.update_segments(self.segments, self.seg_list.currentRow())
	
	def clear_segments (self):
		self.segments.clear()
		self.seg_list.clear()
		self.segs_renamed.clear()
	#	self.seg_list.setCurrentRow(None)
		self.canvas.update_segments(self.segments, self.seg_list.currentRow())
	
	def select_segment (self):
		self.current_segment = self.seg_list.currentItem()
		self.refresh_segments()
	
	def rename_segment (self):
		self.current_segment = self.seg_list.currentRow()
		text, okPressed = QInputDialog.getText(self, 'New name',
						'New name:', text=self.seg_list.currentItem().text())
		if okPressed and text != '':
			self.seg_list.currentItem().setText(text)
			self.segs_renamed[self.seg_list.currentRow()] = True
	
	def select_brush_size (self):
		self.brush_size = self.slider_brush_size.value()
	
	def import_segments (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								'Open Segmentation File', '',
								'PNG Files (*.png);;' + \
								'PTU Files (*.tif);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.tif':
				seg_image = plt.imread(file_path)
				if len(seg_image.shape) == 3:
					seg_image = seg_image[:,:,1]
			elif file_path.suffix.lower() == '.png':
				seg_image = plt.imread(file_path)
				if len(seg_image.shape) == 3:
					seg_image = seg_image[:,:,1]
			else:
				return False
			print(seg_image.shape)
			if self.image_array.shape[0] != seg_image.shape[0] or \
			   self.image_array.shape[1] != seg_image.shape[1]:
				seg_image = cv.resize(seg_image,
									self.image_array[:,:,0].shape,
								#	fx=self.image_array.shape[1]/seg_image[1],
								#	fy=self.image_array.shape[0]/seg_image[0],
									interpolation = cv.INTER_NEAREST)
			self.clear_segments()
			for index, seg_value in enumerate(np.unique(seg_image)):
				self.add_segment()
				self.seg_list.setCurrentRow(index)
				self.segments[index] = (seg_image == seg_value)
				self.select_segment()
	#TODO: make general save/open dialogs - https://coderscratchpad.com/pyqt6-saving-files-with-qfiledialog/
	def export_segments (self): #TODO: add TIF format
		if self.segments == None:
			return False
		if len(self.segments) == 0:
			return False
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getSaveFileName(self,
								'Save File', '',
								'PNG Files (*.png);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		file_path = Path(file_name)
		seg_image = np.zeros_like(self.image_array[:,:,0], dtype = int)
		for index, segment in enumerate(self.segments):
			seg_image[segment] = index+1
		if file_path.suffix.lower() == '.png':
			plt.imsave(file_path,
						seg_image,
						format = 'png')
			return True
		else:
			plt.imsave(file_path.with_suffix('.png'),
						seg_image,
						format = 'png')
			return True
	
	def flip_segments (self):
		if self.segments is None:
			return False
		if len(self.segments) == 0:
			return False
		for index, segment in enumerate(self.segments):
			self.segments[index] = segment[::-1,:]
		self.refresh_segments()
		return True
	
	#TODO: want to be able to get edges of segments
	def find_edges (self):
		if self.segments is None:
			return False
		if len(self.segments) == 0:
			return False
		edges = []
		
	
	def select_bounds (self):
		self.zoomed = False
		self.checkbox_zoom.setChecked(False)
		self.selecting_area = True
#		self.click_id = self.canvas.mpl_connect(
#							'button_press_event', self.on_click)
	
	# mouse interaction with canvas
	def on_click (self, event):
		self.position = np.array([int(np.floor(event.xdata)),
								  int(np.floor(event.ydata))])
		if (self.position[0] < self.x_lower) or \
		   (self.position[0] > self.x_upper) or \
		   (self.position[1] < self.y_lower) or \
		   (self.position[1] > self.y_upper):
			return
		if self.selecting_area:
		#	self.position = np.array([int(np.floor(event.xdata)),
		#							  int(np.floor(event.ydata))])
			self.canvas.mpl_disconnect(self.click_id)
			self.click_id = self.canvas.mpl_connect(
								'button_release_event', self.off_click)
			self.move_id = self.canvas.mpl_connect(
								'motion_notify_event', self.mouse_moved)
		elif event.button is MouseButton.LEFT:
			if self.edit_segments:
				self.canvas.mpl_disconnect(self.click_id)
				self.click_id = self.canvas.mpl_connect(
								'button_release_event', self.off_click)
				self.move_id = self.canvas.mpl_connect(
								'motion_notify_event', self.mouse_moved)
			elif self.use_grid:
				if self.grid_results is not None:
					self.update_fit_plot(
								self.grid_results[
									self.position[1], self.position[0]])
			else:
				if self.segment_results is not None:
					self.update_fit_plot(
								self.segment_results[
									self.position[1], self.position[0]])
#		elif event.button is MouseButton.RIGHT:
#			if self.edit_segments:
#				self.erase_segment = True
#				self.canvas.mpl_disconnect(self.click_id)
#				self.click_id = self.canvas.mpl_connect(
#								'button_release_event', self.off_click)
#				self.move_id = self.canvas.mpl_connect(
#								'motion_notify_event', self.mouse_moved)
	
	def mouse_moved (self, event):
		position = np.array([int(np.floor(event.xdata)),
							 int(np.floor(event.ydata))])
		if self.selecting_area:
			self.canvas.plot_selector(position, self.position)
		elif self.edit_segments:
			seg_index = self.seg_list.currentRow()
			distance = self.brush_size
			if self.seg_list.currentItem() is None:
				return
			elif self.erase_segment:
				self.segments[seg_index][
						position[1]-distance:position[1]+distance+1,
						position[0]-distance:position[0]+distance+1] = False
			else:
				self.segments[seg_index][
						position[1]-distance:position[1]+distance+1,
						position[0]-distance:position[0]+distance+1] = True
			self.canvas.update_segments(self.segments, seg_index)
	
	def off_click (self, event):
		self.canvas.mpl_disconnect(self.click_id)
		self.canvas.mpl_disconnect(self.move_id)
		self.click_id = self.canvas.mpl_connect(
							'button_press_event', self.on_click)
		if self.selecting_area:
			p_1 = np.array([int(np.floor(event.xdata)),
							int(np.floor(event.ydata))])
			p_2 = self.position
			self.canvas.remove_selector()
			self.selecting_area = False
			x_lower = np.amin(np.array([p_1[0], p_2[0]]))
			x_upper = np.amax(np.array([p_1[0], p_2[0]]))
			y_lower = np.amin(np.array([p_1[1], p_2[1]]))
			y_upper = np.amax(np.array([p_1[1], p_2[1]]))
			self.x_lower = x_lower
			self.x_upper = x_upper
			self.y_lower = y_lower
			self.y_upper = y_upper
			self.setup_bound_textboxes()
			self.bound_textbox_select()
	
	def eventFilter(self, source, event):
		if (event.type() == QEvent.ContextMenu and
				source is self.seg_list):
			menu = QMenu()
			Delete = menu.addAction('Delete')
			Rename = menu.addAction('Rename')
			action = menu.exec_(event.globalPos())
		#	action = menu.exec_(self.mapToGlobal(event.pos())) #when inside self
			if action == Delete:
				self.remove_segment()
			elif action == Rename:
				self.rename_segment()
			return True
		return super(Window, self).eventFilter(source, event)
	
	def reset_bounds (self):
		self.x_lower = 0
		self.x_upper = self.x_size
		self.y_lower = 0
		self.y_upper = self.y_size
		self.setup_bound_textboxes()
		self.bound_textbox_select()
	
	def open_file (self):
		if self.file_dialog():
			if self.file_path.suffix.lower() == '.ptu':
				ptu_stream = PTUreader(self.file_path,
										print_header_data = True)
				self.data_stack = ptu_stream.get_flim_data_stack()
				self.image_array = np.sum(self.data_stack[:,:,:,:], axis=3)
				self.grid_heatmap = np.array([[0]], dtype = float)
				self.segment_heatmap = np.array([[0]], dtype = float)
				self.x_size = ptu_stream.head['ImgHdr_PixX']
				self.y_size = ptu_stream.head['ImgHdr_PixY']
				self.xy_res = ptu_stream.head['ImgHdr_PixResol'] #µm
				self.t_res = ptu_stream.head['MeasDesc_Resolution']*10**9 #ns
				self.num_channels = ptu_stream.head['HW_InpChannels']
				time_series = np.sum(self.data_stack,axis=(0,1,2))
				lower_bound = np.amax(time_series > np.amax(time_series)*0.01)
				self.peak_index = np.argmax(time_series) - lower_bound
				self.data_stack = self.data_stack[:,:,:,lower_bound:]
				self.data_stack
				self.channel_box.clear()
				for index in range(self.num_channels):
					self.channel_box.addItem(f'{index:d}')
				self.channel_box.setCurrentIndex = 0
				self.channel = 0
				self.clear_segments()
				self.reset_bounds()
				self.refresh_image()
				photon_count = np.sum(self.image_array)/self.x_size/self.y_size
				self.instruction_text.setText(
						f'X/Y Resolution: {self.xy_res:2.4} µm \t' + \
						f'Time Resolution: {self.t_res:2.4} ns \t' + \
						f'Average Photons: {photon_count:2.4}')
	
	def file_dialog (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								'Open Microscope File', '',
								'PTU Files (*.ptu);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.ptu':
				self.file_path = file_path
				return True
			else:
				self.file_path = None
				return False
	
	def get_fit_function (self, fit_all = False):
		if fit_all:
			fit_function = BEC
			initial_guess = [
						(1-self.af_fraction_guess), self.lifetime_guess,
						self.af_fraction_guess, self.af_life_guess,
						self.irf_centre_guess, self.irf_width_guess ]
		elif self.checkbox_use_af.isChecked():
			if self.fit_each_af:
				fit_function = lambda x, B, tau2, A, tau1: \
								BEC(x, B, tau2, A, tau1,
									self.irf_centre, self.irf_width)
				initial_guess = [
						(1-self.af_fraction_guess), self.lifetime_guess,
						self.af_fraction_guess, self.af_life_guess ]
			else:
				fit_function = lambda x, B, tau2, A: \
								BEC(x, B, tau2, A, self.af_lifetime,
									self.irf_centre, self.irf_width)
				initial_guess = [
						(1-self.af_fraction_guess), self.lifetime_guess,
						self.af_fraction_guess ]
		else:
				fit_function = lambda x, A, tau: \
							MEC(x, A, tau, self.irf_centre, self.irf_width)
				initial_guess = [1.0, self.lifetime_guess]
		return fit_function, initial_guess
	
	def fit_irf (self):
		fit_function, initial_guess = self.get_fit_function(fit_all = True)
		time_points = (np.arange(self.data_stack.shape[-1]) - \
												self.peak_index) * self.t_res
		if self.checkbox_channel.isChecked():
			data_stack = self.data_stack[:,:,self.channel,:]
		else:
			data_stack = np.sum(self.data_stack, axis=2)
		data_points = np.sum(data_stack, axis=(0,1))
		self.full_field_results = fit_data(fit_function, initial_guess,
												time_points, data_points)
		self.af_fraction_guess = self.full_field_results.best_params[2]
		self.af_life_guess = self.full_field_results.best_params[3]
		self.af_lifetime = self.full_field_results.best_params[3]
		self.irf_centre = self.full_field_results.best_params[4]
		self.irf_width = self.full_field_results.best_params[5]
		self.startpoint = self.full_field_results.startpoint
		self.endpoint = self.full_field_results.endpoint
		self.setup_fit_textboxes()
		self.checkbox_fit_each.setChecked(False)
		self.update_fit_plot(self.full_field_results)
	
	def fit_all (self):
		time_points = (np.arange(self.data_stack.shape[-1]) - \
												self.peak_index)*self.t_res
		fit_function, initial_guess = self.get_fit_function()
		if self.checkbox_channel.isChecked():
			data_array = self.data_stack[:,:,self.channel,:]
			photons = self.image_array[:,:,self.channel]
		else:
			data_array = np.sum(self.data_stack, axis=2)
			photons = np.sum(self.image_array, axis=2)
		if self.use_grid:
			self.grid_results = np.empty((self.image_array.shape[0],
								 self.image_array.shape[1]), dtype = object)
			self.fit_grid(time_points, photons, data_array)
		else:
			self.segment_results = np.empty((self.image_array.shape[0],
								 self.image_array.shape[1]), dtype = object)
			self.fit_segments(time_points, photons, data_array)
		self.refresh_image()
	
	def fit_grid (self, time_points, photons, data_array):
		self.grid_heatmap = np.zeros_like(self.image_array[:,:,0],
												dtype = float)
		fit_function, initial_guess = self.get_fit_function()
		photons = photons[self.y_lower:self.y_upper,
						  self.x_lower:self.x_upper]
		data_array = data_array[self.y_lower:self.y_upper,
								self.x_lower:self.x_upper]
		grid_x = int(np.floor(
						(self.x_upper - self.x_lower)/self.grid_factor))
		grid_y = int(np.floor(
						(self.y_upper - self.y_lower)/self.grid_factor))
		update_progress_bar(self.progress_bar, value = 0,
						minimum_value = 0,
						maximum_value = grid_x * grid_y,
						text = 'Fitting Grid Chunks: %p%')
		if self.grid_factor > 1:
			grid_photons = np.sum(window_over(photons, self.grid_factor,
											axes=(0,1)), axis=(2,3))
			grid_array = np.sum(window_over(data_array, self.grid_factor,
											axes=(0,1)), axis=(2,3))
		else:
			grid_photons = photons
			grid_array = data_array
		grid_results = np.empty((grid_y, grid_x), dtype = object)
#		# need array of arguments for parallel execution
#		Y, X = np.meshgrid(np.arange(grid_y), np.arange(grid_x))
#		positions = np.vstack([Y.ravel(), X.ravel()])
#		good_positions = np.zeros(len(positions), dtype = bool)
#		for index, position in enumerate(positions):
#			if grid_photons[position[0], position[1]] > self.photon_threshold:
#				good_positions[index] = True
#		positions = positions[good_positions]
#		data_arguments = np.empty(len(positions), dtype = object)
#		for index, position in enumerate(positions):
#			data_arguments[index] = data_array[position[0], position[1]]
#		with Pool(processes = self.cores_to_use) as pool:
#			parallel_output = pool.map(
#							partial(fit_data, fit_function,
#									initial_guess, time_points),
#							data_arguments)
#		for index, position in enumerate(positions):
#			grid_results[position[0], position[1]] = parallel_output[index]
		for y_index in range(grid_y):
			for x_index in range(grid_x):
				update_progress_bar(self.progress_bar,
						value = y_index * grid_x + x_index)
				if grid_photons[y_index, x_index] < self.photon_threshold:
					continue
		########################################################################
				grid_results[y_index, x_index] = fit_data(
								fit_function, initial_guess,
								time_points, grid_array[y_index, x_index])
		########################################################################
#				data_points = grid_array[y_index, x_index]
#				total_photons = grid_photons[y_index, x_index]
#				peak_index = np.argmax(data_points)
#				peak_photons = data_points[peak_index]
#				data_points = data_points / peak_photons
#				fit = perform_fit(time_points, grid_array[y_index, x_index],
#								  fit_function, initial_guess,
#								  startpoint = self.startpoint,
#								  endpoint = self.endpoint)
#				best_params = fit[0]
#				grid_results[y_index, x_index] = FitResults(
#								fit_function, best_params,
#								time_points, data_points,
#								self.startpoint, self.endpoint,
#								total_photons, peak_photons )
		########################################################################
				lifetime = grid_results[y_index, x_index].best_params[1]
				if lifetime > self.lifetime_max or \
				   lifetime < self.lifetime_min:
					grid_results[y_index,x_index] = None
		update_progress_bar(self.progress_bar, value = 0,
						minimum_value = 0,
						maximum_value = grid_x * grid_y,
						text = 'Generating Heatmap: %p%')
		for y_index in range(grid_y):
			for x_index in range(grid_x):
				update_progress_bar(self.progress_bar,
						value = y_index * grid_x + x_index)
				if grid_results[y_index, x_index] is None:
					continue
				self.grid_results[self.y_lower+self.grid_factor*y_index:
							 self.y_lower+self.grid_factor*(y_index+1),
							 self.x_lower+self.grid_factor*x_index:
							 self.x_lower+self.grid_factor*(x_index+1)] = \
						grid_results[y_index, x_index]
				self.grid_heatmap[self.y_lower+self.grid_factor*y_index:
							 self.y_lower+self.grid_factor*(y_index+1),
							 self.x_lower+self.grid_factor*x_index:
							 self.x_lower+self.grid_factor*(x_index+1)] = \
						grid_results[y_index, x_index].best_params[1]
		self.grid_heatmap[self.grid_heatmap<self.lifetime_min] = 0
		self.grid_heatmap[self.grid_heatmap>self.lifetime_max] = 0
		self.refresh_heatmap()
		self.progress_counter = 0
		clear_progress_bar(self.progress_bar)
	
	def fit_segments (self, time_points, photons, data_array):
		if self.segments is None:
			return
		if len(self.segments) == 0:
			return
		self.segment_heatmap = np.zeros_like(self.image_array[:,:,0],
												dtype = float)
		fit_function, initial_guess = self.get_fit_function()
		update_progress_bar(self.progress_bar, value = 0,
						minimum_value = 0,
						maximum_value = len(self.segments),
						text = 'Fitting Segments: %p%')
		segment_results = np.empty(len(self.segments), dtype = object)
		for index, segment in enumerate(self.segments):
			if np.sum(photons[segment]) < self.photon_threshold:
				continue
			update_progress_bar(self.progress_bar, value = index)
			segment_results[index] = fit_data(
								fit_function, initial_guess,
								time_points,
								np.sum(data_array[segment],axis=0) )
			lifetime = segment_results[index].best_params[1]
			if lifetime > self.lifetime_max or \
			   lifetime < self.lifetime_min:
				segment_results[index] = None
		update_progress_bar(self.progress_bar, value = 0,
						minimum_value = 0,
						maximum_value = len(self.segments),
						text = 'Generating Heatmap: %p%')
		for index, segment in enumerate(self.segments):
			update_progress_bar(self.progress_bar,
								value = index)
			if segment_results[index] is None:
				continue
			self.segment_results[segment] = segment_results[index]
			self.segment_heatmap[segment] = \
								segment_results[index].best_params[1]
		self.segment_heatmap[self.segment_heatmap<self.lifetime_min] = 0
		self.segment_heatmap[self.segment_heatmap>self.lifetime_max] = 0
		self.refresh_heatmap()
		self.progress_counter = 0
		clear_progress_bar(self.progress_bar)
	
	def update_fit_plot (self, results):
		if results is None:
			return False
		self.result_text.setText(
					f'measurement: {results.best_params[1]:7.5f}ns\t' + \
					f'peak photons: {results.peak_photons:6d}\t' + \
					f'total photons: {results.total_photons:6d}\t')
		self.plot_canvas.update_plot(results)
	
	def progress (self):
		self.progress_counter += 1
		update_progress_bar(self.progress_bar, value = self.progress_counter)
	
	def refresh_segments (self):
		if self.show_segments:
			self.canvas.update_segments(self.segments,
										self.seg_list.currentRow())
		else:
			self.canvas.update_segments()
	
	def refresh_image (self):
		if self.checkbox_channel.isChecked():
			self.canvas.update_image(self.image_array[:,:,self.channel])
		else:
			self.canvas.update_image(np.sum(self.image_array, axis=-1))
		self.refresh_heatmap()
	
	def refresh_heatmap (self):
		if self.use_grid:
			self.canvas.update_heatmap(self.grid_heatmap)
		else:
			self.canvas.update_heatmap(self.segment_heatmap)
	
	def save_csv (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getSaveFileName(self,
								'Save File', '',
								'CSV Files (*.csv);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			if self.use_grid:
				heatmap = self.grid_heatmap
			else:
				heatmap = self.segment_heatmap
			if heatmap is None:
				return False
			elif len(heatmap) == 0:
				return False
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.csv':
				np.savetxt(file_path,
							heatmap,
							delimiter = ',')
				return True
			else:
				np.savetxt(file_path.with_suffix('.csv'),
							heatmap,
							delimiter = ',')
				return True
	
	def save_image (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getSaveFileName(self,
								'Save File', '',
								'SVG Files (*.svg);;' + \
								'PNG Files (*.png);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.svg':
				self.canvas.fig.savefig(file_path, format='svg')
				return True
			elif file_path.suffix.lower() == '.png':
				self.canvas.fig.savefig(file_path, format='png')
				return True
			else:
				self.canvas.fig.savefig(file_path.with_suffix('.png'),
										format='png')
				return True

################################################################################

################################################################################

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = Window()
	window.resize(920,1200)
	window.show()
	sys.exit(app.exec_())

################################################################################
# EOF

