import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BeamVisualizer:
    def __init__(self, conn):
        self.conn = conn

    def start_process(self):
        # obtain the generator function for the angle
        # self.yield_angle = yield_angle
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.az = np.radians(np.arange(-180, 180, 1))
        self.num_antennas = 8

        # Initial plot setup
        self.steer_az = np.radians(30)  # Initial steering angle
        weights = 1 / self.get_ula_response(self.num_antennas, self.steer_az)
        g = self.get_ula_gain(self.num_antennas, weights, self.az)
        self.line, = ax.plot(self.az, np.abs(g))

        ax.set_ylim(0, np.max(np.abs(g)) + 1)


        self.steer_az = 30


        # Create the animation
        ani = FuncAnimation(fig, self.update, frames=self.yield_angle, blit=True)

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

    # Update function for animation
    def update(self, frame):
        # print(frame)
        steer_az = np.radians(frame)  # Varying steering angle
        weights = 1 / self.get_ula_response(self.num_antennas, steer_az)
        g = self.get_ula_gain(self.num_antennas, weights, self.az)
        self.line.set_ydata(np.abs(g))
        return self.line,

    # A generator for angles
    def yield_angle(self):
        self.steer_az = self.conn.recv()
        # print(self.steer_az)
        # self.steer_az += 1
        yield self.steer_az % 180

# beam = BeamVisualizer(None)
# beam.start_process()
