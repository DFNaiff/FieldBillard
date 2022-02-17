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

from . import visutils


class Visualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initializerUI()
        
    def initializerUI(self):
        self.setGeometry(100, 100, 900, 600)
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
        point_designs = ["3-Isosceles", "3-Equilateral", "3-Isosceles-B", "3-Random",
                         "4-Cross", "4-Diamond","4-Square", "4-Random",
                         "5-Random", "6-Random", "12-Random", "24-Random"]
        point_title = QLabel("Points:")
        self.point_combobox = QComboBox()
        self.point_combobox.addItems(point_designs)
        design_hbox.addWidget(point_title)
        design_hbox.addWidget(self.point_combobox)

        noise_hbox = QHBoxLayout()
        noise_title = QLabel("Noise")
        self.noise_ledit = QLineEdit()
        self.noise_ledit.setText("0.0")
        noise_hbox.addWidget(noise_title)
        noise_hbox.addWidget(self.noise_ledit)
        
        frame_hbox = QHBoxLayout()
        frame_designs = ["Circle", "Hash", "Square"]
        frame_title = QLabel("Frame:")
        self.frame_combobox = QComboBox()
        self.frame_combobox.addItems(frame_designs)
        frame_hbox.addWidget(frame_title)
        frame_hbox.addWidget(self.frame_combobox)
        
        mass_hbox = QHBoxLayout()
        mass_label = QLabel("Mass:")
        self.mass_ledit = QLineEdit()
        self.mass_ledit.setText("1.0")
        mass_hbox.addWidget(mass_label)
        mass_hbox.addWidget(self.mass_ledit)

        charge_hbox = QHBoxLayout()
        charge_label = QLabel("Charge:")
        self.charge_ledit = QLineEdit()
        self.charge_ledit.setText("1.0")
        charge_hbox.addWidget(charge_label)
        charge_hbox.addWidget(self.charge_ledit)

        frame_charge_hbox = QHBoxLayout()
        frame_charge_label = QLabel("Frame charge:")
        self.frame_charge_ledit = QLineEdit()
        self.frame_charge_ledit.setText("10.0")
        frame_charge_hbox.addWidget(frame_charge_label)
        frame_charge_hbox.addWidget(self.frame_charge_ledit)

        magnetics_hbox = QHBoxLayout()
        self.magnetics_checkbox = QCheckBox("Magnetics")
        magnetics_title = QLabel("Coupling")
        self.magnetics_ledit = QLineEdit()
        self.magnetics_ledit.setText("0.01")
        magnetics_hbox.addWidget(self.magnetics_checkbox)
        magnetics_hbox.addWidget(magnetics_title)
        magnetics_hbox.addWidget(self.magnetics_ledit)

        integrator_hbox = QHBoxLayout()
        integrator_designs = ["SympleticEuler", "SympleticVerlet",
                              "Euler", "Midpoint", "RungeKutta"]
        integrator_title = QLabel("Integrator:")
        self.integrator_combobox = QComboBox()
        self.integrator_combobox.addItems(integrator_designs)
        integrator_hbox.addWidget(integrator_title)
        integrator_hbox.addWidget(self.integrator_combobox)
        
        timestep_hbox = QHBoxLayout()
        timestep_title = QLabel("Step")
        self.timestep = QLineEdit()
        self.timestep.setText("0.01")
        render_interval_text = QLabel("Render")
        self.render_ledit = QLineEdit()
        self.render_ledit.setText("1")
        timestep_hbox.addWidget(timestep_title)
        timestep_hbox.addWidget(self.timestep)
        timestep_hbox.addWidget(render_interval_text)
        timestep_hbox.addWidget(self.render_ledit)
        
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
        
        self.layout.addLayout(design_hbox)
        self.layout.addLayout(noise_hbox)
        self.layout.addLayout(frame_hbox)
        self.layout.addLayout(mass_hbox)
        self.layout.addLayout(charge_hbox)
        self.layout.addLayout(frame_charge_hbox)
        self.layout.addLayout(magnetics_hbox)
        self.layout.addLayout(integrator_hbox)
        self.layout.addLayout(timestep_hbox)
        self.layout.addLayout(memory_hbox)
        self.layout.addWidget(create_button)
        self.layout.addWidget(run_button)
        self.layout.addWidget(snap_button)
        self.setLayout(self.layout)
        
        frame_rate = 30 #TODO: put something better        
        self.timer = QTimer()
        self.timer.setInterval(int(1/frame_rate)*1000)
        self.timer.timeout.connect(self.update)

    def create(self):
        self.timer.stop()
        self.parent.plot.reset_plot()
        point_design = self.point_combobox.currentText()
        frame_design = self.frame_combobox.currentText()
        integrator = self.integrator_combobox.currentText()
        try:
            charge = float(self.charge_ledit.text())
            frame_charge = float(self.frame_charge_ledit.text())
            mass = float(self.mass_ledit.text())
            noise = float(self.noise_ledit.text())
            magnetic_coupling = None if not self.magnetics_checkbox.isChecked()\
                                else float(self.magnetics_ledit.text())
            self.memory_size = int(self.memory_ledit.text())
            self.dt = float(self.timestep.text())
            self.nrender = int(self.render_ledit.text())
            self.t = 0.0
            assert charge >= 0
            assert frame_charge > 0
            assert mass > 0
            assert noise >= 0
            assert (True if magnetic_coupling is None else magnetic_coupling >= 0)
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
                                                         magnetic_coupling)
        visutils.set_system_frame(self.system, frame_design, frame_charge)
        visutils.set_integrator(self.system, integrator)
        if self.has_memory:
            self.memory = visutils.collections.deque([], maxlen=self.memory_size)
            self.memory.append(self.system.points.xy.detach().numpy())
        self.parent.plot.init_scatter(self.system)

    def run(self):
        assert hasattr(self, "system")
        self.timer.start()

    def update(self):
        try:
            for i in range(self.nrender):
                self.system.step(self.dt)
        except visutils.integrators.NonValidIntegratorError:
            QMessageBox.critical(self, 
                                 "Could not run system",
                                 "Could not run system. Probably non-compatible integrator",
                                 QMessageBox.Close,
                                 QMessageBox.Close)
            self.timer.stop()
        if self.has_memory:
            self.memory.append(self.system.points.xy.detach().numpy())
        self.parent.plot.update_scatter(self.system, None)
        self.t += self.dt

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