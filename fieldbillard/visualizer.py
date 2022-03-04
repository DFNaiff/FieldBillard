# -*- coding: utf-8 -*-
import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow,
                             QHBoxLayout, QVBoxLayout,
                             QComboBox, QLabel, QPushButton,
                             QLineEdit, QCheckBox, QFileDialog,
                             QMessageBox)
from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import torch

from . import visutils


POINTS_DESIGNS = \
    ["N-Random-Circle", "N-Random-Square", "N-Equilateral", 
     "3-Isosceles", "3-Equilateral", "3-Isosceles-B",
     "4-Cross", "4-Diamond","4-Square"]
FRAMES_DESIGNS = \
    ["Circle", "Hash", "Square", "Periodic", "XPeriodic", "YPeriodic"]
FIXED_POINTS_DESIGNS = ["None", "RandomCircle", "RandomSquare"]
INTEGRATORS = \
    ["SympleticEuler", "SympleticVerlet",
     "Tao20", "Tao80", "Tao320"] 

LICENSE_MESSAGE = \
"""\
BSD 3-Clause License

Copyright (c) 2022, Danilo de Freitas Naiff.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

class Visualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initializerUI()
        
    def initializerUI(self):
        self.setGeometry(100, 100, 1000, 600)
        self.setWindowTitle("Field Billard")
        self.setCentralWidget(CentralWidget(self))
        self.show()
        

class CentralWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initializeUI()
        
    def initializeUI(self):
        self.main_layout = QHBoxLayout(self)
        self.form = FormWidget(self)
        self.plot = PlotWidget(self)
        self.main_layout.addWidget(self.form)
        self.main_layout.addWidget(self.plot)
        self.setLayout(self.main_layout)


class FormWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initializeUI()

    def initializeUI(self):
        self.layout = QVBoxLayout()

        design_hbox = QHBoxLayout()        
        point_designs = POINTS_DESIGNS
        point_title = QLabel("Points:")
        self.point_combobox = QComboBox()
        self.point_combobox.addItems(point_designs)
        radius_title = QLabel("R")
        self.radius_ledit = QLineEdit()
        self.radius_ledit.setText("1.0")
        npoints_title = QLabel("R")
        self.npoints_ledit = QLineEdit()
        self.npoints_ledit.setText("4")        
        noise_title = QLabel("Noise")
        self.noise_ledit = QLineEdit()
        self.noise_ledit.setText("0.0")
        design_hbox.addWidget(point_title)
        design_hbox.addWidget(self.point_combobox)
        design_hbox.addWidget(radius_title)
        design_hbox.addWidget(self.radius_ledit)
        design_hbox.addWidget(npoints_title)
        design_hbox.addWidget(self.npoints_ledit)
        design_hbox.addWidget(noise_title)
        design_hbox.addWidget(self.noise_ledit)

        frame_hbox = QHBoxLayout()
        frame_designs = FRAMES_DESIGNS
        frame_title = QLabel("Frame:")
        self.frame_combobox = QComboBox()
        self.frame_combobox.addItems(frame_designs)
        frame_charge_label = QLabel("Intensity:")
        self.frame_charge_ledit = QLineEdit()
        self.frame_charge_ledit.setText("10.0")
        frame_hbox.addWidget(frame_title)
        frame_hbox.addWidget(self.frame_combobox)
        frame_hbox.addWidget(frame_charge_label)
        frame_hbox.addWidget(self.frame_charge_ledit)
        
        fixed_points_hbox = QHBoxLayout()
        fixed_points_designs = FIXED_POINTS_DESIGNS
        fixed_points_name = QLabel("FPoints")
        self.fixed_points_combobox = QComboBox()
        self.fixed_points_combobox.addItems(fixed_points_designs)
        fixed_points_number_name = QLabel("N:")
        self.fixed_points_number_ledit = QLineEdit()
        self.fixed_points_number_ledit.setText("3")
        fixed_points_charge_label = QLabel("Q:")
        self.fixed_points_charge_ledit = QLineEdit()
        self.fixed_points_charge_ledit.setText("1.0")
        fixed_points_hbox.addWidget(fixed_points_name)
        fixed_points_hbox.addWidget(self.fixed_points_combobox)
        fixed_points_hbox.addWidget(fixed_points_number_name)
        fixed_points_hbox.addWidget(self.fixed_points_number_ledit)
        fixed_points_hbox.addWidget(fixed_points_charge_label)
        fixed_points_hbox.addWidget(self.fixed_points_charge_ledit)
        
        mass_hbox = QHBoxLayout()
        mass_label = QLabel("Mass:")
        self.mass_ledit = QLineEdit()
        self.mass_ledit.setText("1.0")
        charge_label = QLabel("Charge:")
        self.charge_ledit = QLineEdit()
        self.charge_ledit.setText("1.0")
        mass_hbox.addWidget(mass_label)
        mass_hbox.addWidget(self.mass_ledit)
        mass_hbox.addWidget(charge_label)
        mass_hbox.addWidget(self.charge_ledit)

        darwin_hbox = QHBoxLayout()
        self.darwin_checkbox = QCheckBox("Darwin")
        darwin_title = QLabel("Coupling")
        self.darwin_ledit = QLineEdit()
        self.darwin_ledit.setText("0.01")
        darwin_hbox.addWidget(self.darwin_checkbox)
        darwin_hbox.addWidget(darwin_title)
        darwin_hbox.addWidget(self.darwin_ledit)

        integrator_hbox = QHBoxLayout()
        integrator_designs = INTEGRATORS
        integrator_title = QLabel("Integrator:")
        self.integrator_combobox = QComboBox()
        self.integrator_combobox.addItems(integrator_designs)
        self.integrator_combobox.setCurrentIndex(1)
        timestep_title = QLabel("Step")
        self.timestep = QLineEdit()
        self.timestep.setText("0.001")
        render_interval_text = QLabel("Render")
        self.render_ledit = QLineEdit()
        self.render_ledit.setText("10")
        integrator_hbox.addWidget(integrator_title)
        integrator_hbox.addWidget(self.integrator_combobox)
        integrator_hbox.addWidget(timestep_title)
        integrator_hbox.addWidget(self.timestep)
        integrator_hbox.addWidget(render_interval_text)
        integrator_hbox.addWidget(self.render_ledit)
        
        # timestep_hbox = QHBoxLayout()
        # timestep_title = QLabel("Step")
        # self.timestep = QLineEdit()
        # self.timestep.setText("0.001")
        # render_interval_text = QLabel("Render")
        # self.render_ledit = QLineEdit()
        # self.render_ledit.setText("10")
        # timestep_hbox.addWidget(timestep_title)
        # timestep_hbox.addWidget(self.timestep)
        # timestep_hbox.addWidget(render_interval_text)
        # timestep_hbox.addWidget(self.render_ledit)
        
        memory_hbox = QHBoxLayout()
        self.memory_checkbox = QCheckBox("Memory")
        memory_title = QLabel("Size")
        self.memory_ledit = QLineEdit()
        self.memory_ledit.setText("50")
        memory_hbox.addWidget(self.memory_checkbox)
        memory_hbox.addWidget(memory_title)
        memory_hbox.addWidget(self.memory_ledit)
        
        create_button = QPushButton("Create")
        create_button.clicked.connect(self.create)
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run)
        snap_button = QPushButton("Snap")
        snap_button.clicked.connect(self.snap)
        license_button = QPushButton("License")
        license_button.clicked.connect(self.show_license)
        
        self.layout.addLayout(design_hbox)
        self.layout.addLayout(frame_hbox)
        self.layout.addLayout(fixed_points_hbox)
        self.layout.addLayout(mass_hbox)
        self.layout.addLayout(darwin_hbox)
        self.layout.addLayout(integrator_hbox)
        self.layout.addLayout(memory_hbox)
        self.layout.addWidget(create_button)
        self.layout.addWidget(run_button)
        self.layout.addWidget(snap_button)
        self.layout.addWidget(license_button)
        self.layout.addStretch(1)
        self.setLayout(self.layout)
        
        frame_rate = 30 #TODO: put something better        
        self.timer = QTimer()
        self.timer.setInterval(int(1/frame_rate)*1000)
        self.timer.timeout.connect(self.update)
    
    def show_license(self):
        QMessageBox.information(self, "License", 
                                LICENSE_MESSAGE, QMessageBox.Ok)
        
    def create(self):
        self.timer.stop()
        self.parent.plot.reset_plot()
        point_design = self.point_combobox.currentText()
        frame_design = self.frame_combobox.currentText()
        fixed_points_design = self.fixed_points_combobox.currentText()
        integrator = self.integrator_combobox.currentText()
        try:
            charge = float(self.charge_ledit.text())
            frame_charge = float(self.frame_charge_ledit.text())
            mass = float(self.mass_ledit.text())
            pradius = float(self.radius_ledit.text())
            npoints = int(self.npoints_ledit.text())
            noise = float(self.noise_ledit.text())
            darwin_coupling = None if not self.darwin_checkbox.isChecked()\
                                else float(self.darwin_ledit.text())
            fixed_point_number = int(self.fixed_points_number_ledit.text())
            fixed_point_charge = float(self.fixed_points_charge_ledit.text())
            self.memory_size = int(self.memory_ledit.text())
            self.dt = float(self.timestep.text())
            self.nrender = int(self.render_ledit.text())
            assert charge >= 0
            assert frame_charge > 0
            assert mass > 0
            assert noise >= 0
            assert (True if darwin_coupling is None else darwin_coupling >= 0)
            assert self.memory_size >= 0
            assert self.dt > 0.0
            assert self.nrender >= 1
        except ValueError:
            QMessageBox.critical(self, 
                                 "Could not start system",
                                 "Could not set parameters. Check the form",
                                 QMessageBox.Close,
                                 QMessageBox.Close)
        self.has_memory = self.memory_checkbox.isChecked()
        self.system = visutils.create_system_from_design(point_design,
                                                         noise,
                                                         mass,
                                                         charge,
                                                         pradius,
                                                         npoints,
                                                         darwin_coupling)
        visutils.set_system_frame(self.system, frame_design, frame_charge)
        fixedx, fixedy = visutils.set_fixed_points(self.system, fixed_points_design,
                                                   fixed_point_number, fixed_point_charge)
        visutils.set_integrator(self.system, integrator)
        if self.has_memory:
            self.memory = visutils.collections.deque([], maxlen=self.memory_size)
            self.memory.append(self.system.points.xy.detach().numpy())
        self.parent.plot.init_scatter(self.system)
        self.draw_objects(frame_design)
        self.parent.plot.draw_points(fixedx, fixedy)
        
    def run(self):
        assert hasattr(self, "system")
        self.timer.start()

    def update(self):
        for i in range(self.nrender):
            try:
                self.system.step(self.dt)
            except visutils.integrators.NonValidIntegratorError:
                QMessageBox.critical(self, 
                                     "Could not run system",
                                     "Non-compatible integrator",
                                     QMessageBox.Close,
                                     QMessageBox.Close)
                self.timer.stop()
                break
        if self.has_memory:
            self.memory.append(self.system.points.xy.detach().clone().numpy())
        self.parent.plot.update_scatter(self.system, None)
        fartest = torch.max(torch.abs(self.system.points.xy.detach())).item()
        if fartest > 2.0:
            QMessageBox.critical(self, "Stopping simulation",
                                 "There are particles out of bounds. "\
                                 "Try a lower time step or a sympletic integrator.",
                                 QMessageBox.Close, QMessageBox.Close)
            self.timer.stop()

    def snap(self):
        self.timer.stop()
        if not self.has_memory:
            return
        else:
            memory_array = np.stack(self.memory, axis=-1)
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save File',
                "","NPY file (*.npy)")
            try:
                np.save(file_name, memory_array)
            except:
                QMessageBox.information(self, "Error", 
                    "Unable to save file.", QMessageBox.Ok)
    
    def draw_objects(self, frame_design):
#"Circle", "Hash", "Square", "Periodic", "XPeriodic", "YPeriodic"
        if frame_design == "Circle":
            self.parent.plot.draw_circle()
        elif frame_design in ["Hash", "Square"]:
            self.parent.plot.draw_square()
        elif frame_design in ["XPeriodic"]:
            self.parent.plot.draw_horizontal_lines()
        elif frame_design in ["YPeriodic"]:
            self.parent.plot.draw_vertical_lines()

            
class PlotWidget(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.reset_plot()
        super().__init__(fig)
        
    def reset_plot(self):
        self.axes.cla()
        self.axes.set_xlim(-1.0, 1.0)
        self.axes.set_ylim(-1.0, 1.0)
    
    def draw_circle(self):
        theta = np.linspace(0, 2*np.pi, 101)
        x = np.cos(theta)
        y = np.sin(theta)
        self.axes.plot(x, y, color='blue')
        self.draw()

    def draw_square(self):
        x = [1.0, -1.0, -1.0, 1.0, 1.0]
        y = [1.0, 1.0, -1.0, -1.0, 1.0]
        self.axes.plot(x, y, color='blue')
        self.draw()

    def draw_vertical_lines(self):
        self.axes.plot([1.0, 1.0], [-1.0, 1.0], color='blue')
        self.axes.plot([-1.0, -1.0], [-1.0, 1.0], color='blue')
        self.draw()

    def draw_horizontal_lines(self):
        self.axes.plot([-1.0, 1.0], [1.0, 1.0], color='blue')
        self.axes.plot([-1.0, 1.0], [-1.0, -1.0], color='blue')
        self.draw()
        
    def draw_points(self, x, y):
        self.axes.scatter(x, y, color='darkblue')
        self.draw()
        
    def init_scatter(self, system):
        self.scatter = self.axes.scatter(system.points.x.detach().numpy(),
                                         system.points.y.detach().numpy(),
                                         color='black')
        #self.title = self.axes.set_title("t = %f"%t)
        self.draw()
    
    def init_scatter_with_memory(self, memory):
        self.axes.cla()
        self.axes.set_xlim(-1.0, 1.0)
        self.axes.set_ylim(-1.0, 1.0)
        for xy, alpha in memory.iterate_with_alpha():
            self.axes.scatter(xy[:, 0], xy[:, 1], color='black', alpha=alpha)
        self.draw()
        
    def update_scatter(self, system, memory=None):
        #self.init_scatter(system, t)
        if memory is None:
            self.scatter.set_offsets(system.points.xy.detach().numpy())
            self.draw()
        else: #Dumb way for doing this
            self.init_scatter_with_memory(memory)
            
            
def run():
    app = QApplication(sys.argv)
    window = Visualizer()
    sys.exit(app.exec_())