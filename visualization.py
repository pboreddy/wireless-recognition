import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

class BeamVisualizer:
    def __init__(self, q):
        self.queue = q
        self.angles = {}
        self.lines = {}
        self.colors = cm.rainbow(np.linspace(0, 1, 10))  # Predefined colors for up to 10 objects

    def start_process(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.ax = ax
        self.az = np.radians(np.arange(-90, 90, 1))
        self.ax.set_thetamin(-90)
        self.ax.set_thetamax(90)
        self.num_antennas = 8

        self.ax.set_ylim(0, 8)  # Adjust as needed based on expected gain values
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)

        # Create the animation
        ani = FuncAnimation(fig, self.update, frames=self.yield_angles, interval=30, blit=True, save_count=1)

        # Display the animation
        plt.show()

    def get_ula_response(self, N, az):
        antennas = np.arange(0, N, 1, dtype='complex128')
        out = np.exp(-np.pi * np.sin(az) * 1j * antennas)
        return out

    def get_ula_gain_indiv(self, N, w, az):
        response = self.get_ula_response(N, az)
        return np.dot(w, response)

    def get_ula_gain(self, N, w, az):
        vget_ula_gain = np.vectorize(self.get_ula_gain_indiv, excluded=["N", "w"])
        return vget_ula_gain(N=N, w=w, az=az)

    def update(self, frame):
        angles = frame
        try:
            for obj_id, (steer_az, obj_cls) in angles.items():
                if obj_id not in self.lines:
                    line, = self.ax.plot(self.az, np.zeros_like(self.az), color=self.colors[obj_cls % len(self.colors)])
                    self.lines[obj_id] = line
                self.lines[obj_id].set_visible(True)
                steer_az_rad = np.radians(steer_az)
                weights = 1 / self.get_ula_response(self.num_antennas, steer_az_rad)
                g = self.get_ula_gain(self.num_antennas, weights, self.az)
                self.lines[obj_id].set_ydata(np.abs(g))
            for obj_id in list(self.lines):
                if obj_id not in angles:
                    self.lines[obj_id].set_visible(False)
                    # could delete the line to save memory
            return [line for line in self.lines.values() if line.get_visible()]
        except:
            return None

    def yield_angles(self):
        while self.queue.empty():
            continue
        # angles maps object ids to angles
        angles = self.queue.get()
        yield angles
