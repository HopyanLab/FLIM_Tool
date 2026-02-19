#!/usr/bin/env python3

import sys
import time
import copy
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import (
					FigureCanvasQTAgg as FigureCanvas,
					NavigationToolbar2QT as NavigationToolbar
							)
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib import colors, ticker, cm
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
from pathlib import Path
from readPTU_FLIM import PTUreader
import pickle as pkl
import gzip
import lzma

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
	try:
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
	except Exception as error:
		message = "An error occurred:"+type(error).__name__+"–"+str(error)
		display_error(message)

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
# wrapper class for image array #
#################################

class DataFLIM ():
	def __init__ (self, data_array, resolution_xy = 1, resolution_t = 1):
		self.data_array = data_array
		self.resolution_xy = resolution_xy
		self.resolution_t = resolution_t

################################################################################
# matplotlib canvas widget #
############################

class MPLCanvas(FigureCanvas):
	def __init__ (self, parent=None, width=10, height=8, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.subplots(1,1)
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		self.fig.set_tight_layout(True)
		#
		self.lines = None
		self.image = None
	
	def plot_lines (self, x_positions, y_positions):
		self.remove_plot_element(self.lines)
		self.lines = None
		if x_positions is None or y_positions is None:
			return False
		self.lines = []
		for x_position in x_positions:
			self.lines.append(self.ax.plot(
										[x_position, x_position],
										[0, y_positions[-1]],
										marker = '',
										linestyle = '-',
										color = 'white',
										zorder = 7))
		for y_position in y_positions:
			self.lines.append(self.ax.plot(
										[0, x_positions[-1]],
										[y_position, y_position],
										marker = '',
										linestyle = '-',
										color = 'white',
										zorder = 7))
		self.ax.set_xlim(left = x_positions[0], right = x_positions[-1])
		self.ax.set_ylim(bottom = x_positions[0], top = y_positions[-1])
		self.draw()
	
	def plot_image (self, image_array = None):
		self.remove_plot_element(self.image)
		self.image = None
		if image_array is None:
			return False
		self.image = self.ax.imshow(image_array,
									cmap = 'Greys_r',
									zorder = 6)
		self.draw()
	
	def clear_canvas (self):
		self.remove_plot_element(self.lines)
		self.remove_plot_element(self.image)
		self.lines = None
		self.image = None
		self.ax.clear()
		self.draw()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for sub_element in plot_element:
					self.remove_plot_element(sub_element)
			else:
				try:
					plot_element.remove()
				except:
					pass


################################################################################
# main window object #
######################

class Window(QWidget):
	def __init__ (self):
		super().__init__()
		self.title = "PTU Stitching Tool"
		self.canvas = MPLCanvas()
		self.toolbar = NavigationToolbar(self.canvas, self)
		#
		self.initialise_variables()
		#
		self.setup_gui()
	
	def initialise_variables (self):
		self.data_arrays = []
		self.data_order = []
		self.res_xy = 1
		self.res_t =1
		#
		self.grid_x = 2
		self.grid_y = 2
		self.fits = False
		#
		self.size_x = 0
		self.size_y = 0
		self.size_t = 0
		self.num_channels = 0
		self.channel = 0
		#
		self.overlap_x = 0
		self.overlap_y = 0
		self.selected = None
	
	def setup_gui (self):
		self.setWindowTitle(self.title)
		main_layout = QVBoxLayout()
		main_layout.addWidget(self.canvas)
		self.canvas.resize(self.canvas.sizeHint())
		toolbar_layout = QHBoxLayout()
		toolbar_layout.addWidget(self.toolbar)
		self.channel_box = setup_combobox(
							self.select_channel,
							toolbar_layout, 'Channel:')
		main_layout.addLayout(toolbar_layout)
		info_box, self.info_text = setup_labelbox(
						'<font color="red">Info: </font>',
						'Open files to begin.')
		main_layout.addWidget(info_box)
		options_layout = self.setup_options_layout()
		main_layout.addLayout(options_layout)
		self.setLayout(main_layout)
		#
		self.click_id = self.canvas.mpl_connect(
							'button_press_event', self.on_click)
	
	def setup_options_layout (self):
		options_layout = QHBoxLayout()
		self.button_open_files = setup_button(
							self.open_files,
							options_layout, 'Choose Dir')
		self.textbox_grid_x = setup_textbox(
							self.textbox_select,
							options_layout, 'X Grid:',
							initial_value = self.grid_x)
		self.textbox_grid_y = setup_textbox(
							self.textbox_select,
							options_layout, 'Y Grid:',
							initial_value = self.grid_y)
		self.textbox_overlap_x = setup_textbox(
							self.textbox_select,
							options_layout, 'X Overlap:',
							initial_value = self.overlap_x)
		self.textbox_overlap_y = setup_textbox(
							self.textbox_select,
							options_layout, 'Y Overlap:',
							initial_value = self.overlap_y)
		self.button_save_pkl = setup_button(
							self.save_pkl,
							options_layout, 'Save PKL')
		#
		return options_layout
	
	def textbox_select (self):
		self.grid_x = get_textbox(self.textbox_grid_x,
									minimum_value = 0,
									maximum_value = 6,
									is_int = True)
		self.grid_y = get_textbox(self.textbox_grid_y,
									minimum_value = 0,
									maximum_value = 6,
									is_int = True)
		self.overlap_x = get_textbox(self.textbox_overlap_x,
									minimum_value = 0,
									maximum_value = 512,
									is_int = True)
		self.overlap_y = get_textbox(self.textbox_overlap_y,
									minimum_value = 0,
									maximum_value = 512,
									is_int = True)
		self.update()
	
	def select_channel (self):
		self.channel = self.channel_box.currentIndex()
		self.update()
	
	def open_files (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								'Open Microscope PTU Files', '',
								'PTU Files (*.ptu);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		file_path = Path(file_name)
		if file_path.suffix.lower() == '.ptu':
			dir_path = file_path.parent
		elif file_path.is_dir():
			dir_path = file_path
		else:
			self.info_text.setText(
					'Selected file was not a PTU file. ' + \
					'Open files to begin.')
			return False
		self.size_t = 0
		self.data_arrays = []
		self.data_order = []
		for index, ptu_file in enumerate(sorted(dir_path.glob('*.ptu'),
											key=lambda x: x.name.lower())):
			ptu_stream = PTUreader(ptu_file, print_header_data = False)
			data_array = ptu_stream.get_flim_data_stack()
			self.data_arrays.append(data_array)
			self.data_order.append(index)
			if index == 0:
				self.resolution_xy = ptu_stream.head['ImgHdr_PixResol'] #µm
				self.resolution_t = ptu_stream.head['MeasDesc_Resolution'] * \
																	10**9 #ns
				self.size_x = ptu_stream.head['ImgHdr_PixX']
				self.size_y = ptu_stream.head['ImgHdr_PixY']
				self.num_channels = ptu_stream.head['HW_InpChannels']
				self.channel_box.clear()
				for index in range(self.num_channels):
					self.channel_box.addItem(f'{index:d}')
				self.channel_box.setCurrentIndex = 0
				self.channel = 0
			self.size_t = np.amax([self.size_t, data_array.shape[3]])
		self.update()
	
	def save_pkl (self):
		if not self.fits:
			return False
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getSaveFileName(self,
								'Save Data to PKL File', '',
								'PKL Files (*.pkl);;' + \
								'GZIP Files (*.gz);;' + \
								'LZMA Files (*.xz);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		file_path = Path(file_name)
		if file_path.suffix.lower() == '.pkl':
			open_function = open
		elif file_path.suffix.lower() == '.gz':
			open_function = gzip.open
		elif file_path.suffix.lower() == '.xz':
			open_function = lzma.open
		else:
			file_path = file_path.with_suffix('.pkl')
			open_function = open
		full_data_array = self.assemble_data_array()
		if full_data_array is None:
			return False
		data_object = DataFLIM(full_data_array, self.resolution_xy,
												self.resolution_t)
		with open_function(file_path, 'wb') as output_file:
			pkl.dump(data_object, output_file)
	
	def assemble_data_array (self):
		if not self.fits:
			return None
		full_data_array = np.zeros((
				self.size_y*self.grid_y - (self.grid_y-1)*self.overlap_y,
				self.size_x*self.grid_x - (self.grid_x-1)*self.overlap_x,
				self.num_channels,
				self.size_t), dtype = int)
		for y in range(self.grid_y):
			for x in range(self.grid_x):
				start_x = x*self.size_x - x*self.overlap_x
				end_x = (x+1)*self.size_x - x*self.overlap_x
				start_y = y*self.size_y - y*self.overlap_y
				end_y = (y+1)*self.size_y - y*self.overlap_y
		#		print(f'{start_x:d}:{end_x:d} {start_y:d}:{end_y:d} ' + \
		#				f'{self.data_order[y*self.grid_x + x]:d}')
				size_t = self.data_arrays[
							self.data_order[y*self.grid_x + x]].shape[3]
				full_data_array[start_y:end_y,
								start_x:end_x,
								:,:size_t] = \
					self.data_arrays[self.data_order[y*self.grid_x + x]]
		return full_data_array
	
	def update (self):
		self.update_info()
		self.update_image()
	
	def update_info (self):
		if self.data_arrays is None or len(self.data_arrays) == 0:
			self.info_text.setText(
							'No data files loaded. Open files to begin.')
			self.fits = False
			return False
		else:
			self.info_text.setText(
							f'Loaded Files: {len(self.data_arrays):d} ' + \
							f'Grid Spaces: {self.grid_x*self.grid_y:d}')
			if len(self.data_arrays) != self.grid_x * self.grid_y:
				self.fits = False
			else:
				self.fits = True
	
	def update_image (self):
		if not self.fits:
			return False
		full_data_array = self.assemble_data_array()
		if full_data_array is None:
			return False
		self.canvas.clear_canvas()
		self.canvas.plot_image(
						np.sum(full_data_array[:,:,self.channel,:],axis=2))
		self.canvas.plot_lines(
						np.arange(self.grid_x+1)*(self.size_x-self.overlap_x),
						np.arange(self.grid_y+1)*(self.size_y-self.overlap_y))
	
	# mouse interaction with canvas
	def on_click (self, event):
		self.position = np.array([int(np.floor(event.xdata)),
								  int(np.floor(event.ydata))])
		if (self.position[0] < 0) or \
		   (self.position[0] > self.size_x*self.grid_x) or \
		   (self.position[1] < 0) or \
		   (self.position[1] > self.size_y*self.grid_y):
			return False
		tile_x = int(np.floor(self.position[0]/(self.size_x)))
		tile_y = int(np.floor(self.position[1]/(self.size_y)))
		tile_index = tile_y*self.grid_x + tile_x
		if self.selected is None:
			self.selected = tile_index
		#	print(f'beep: {tile_index:d}')
		else:
			self.data_order[self.selected], self.data_order[tile_index] = \
				self.data_order[tile_index], self.data_order[self.selected]
			self.selected = None
			self.update()
		#	print(f'boop: {tile_index:d}')


################################################################################

################################################################################

if __name__ == "__main__":
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
	app = QApplication(sys.argv)
	window = Window()
	window.resize(920,1200)
	window.show()
	sys.exit(app.exec_())

################################################################################
# EOF
