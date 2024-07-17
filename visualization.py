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
        self.az = np.radians(np.arange(-180, 180, 1))
        self.num_antennas = 8

        ax.set_ylim(0, 8)  # Adjust as needed based on expected gain values

        # Create the animation
        ani = FuncAnimation(fig, self.update, frames=self.yield_angles, interval=30, blit=True)

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
        for obj_id, steer_az in angles.items():
            if obj_id not in self.lines:
                line, = self.ax.plot(self.az, np.zeros_like(self.az), color=self.colors[obj_id % len(self.colors)])
                self.lines[obj_id] = line
            self.lines[obj_id].set_visible(True)
            steer_az_rad = np.radians(steer_az)
            weights = 1 / self.get_ula_response(self.num_antennas, steer_az_rad)
            g = self.get_ula_gain(self.num_antennas, weights, self.az)
            self.lines[obj_id].set_ydata(np.abs(g))
        for obj_id in list(self.lines):
            if obj_id not in angles:
                self.lines[obj_id].set_visible(False)
        return [line for line in self.lines.values() if line.get_visible()]

    def yield_angles(self):
        while self.queue.empty():
            continue
        # angles maps object ids to angles
        angles = self.queue.get()
        yield angles

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# class BeamVisualizer:
#     def __init__(self, q):
#         self.queue = q
#
#     def start_process(self):
#         # obtain the generator function for the angle
#         # self.yield_angle = yield_angle
#         fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#         self.az = np.radians(np.arange(-180, 180, 1))
#         self.num_antennas = 8
#
#         # Initial plot setup
#         self.steer_az = np.radians(30)  # Initial steering angle
#         weights = 1 / self.get_ula_response(self.num_antennas, self.steer_az)
#         g = self.get_ula_gain(self.num_antennas, weights, self.az)
#         self.line, = ax.plot(self.az, np.abs(g))
#
#         ax.set_ylim(0, np.max(np.abs(g)) + 1)
#
#         self.steer_az = 30
#
#         # Create the animation
#         ani = FuncAnimation(fig, self.update, frames=self.yield_angle, interval=30, blit=True)
#
#         # Display the animation
#         plt.show()
#
#     def get_ula_response(self, N, az):
#         antennas = np.arange(0, N, 1, dtype='complex128')
#         out = np.exp(-np.pi * np.sin(az) * 1j * antennas)
#         return out
#
#     def get_ula_gain_indiv(self, N, w, az):
#         response = self.get_ula_response(N, az)
#         return np.dot(w, response)
#
#     def get_ula_gain(self, N, w, az):
#         vget_ula_gain = np.vectorize(self.get_ula_gain_indiv, excluded=["N", "w"])
#         return vget_ula_gain(N=N, w=w, az=az)
#
#     # Update function for animation
#     def update(self, frame):
#         # print(frame)
#         steer_az = np.radians(frame)  # Varying steering angle
#         weights = 1 / self.get_ula_response(self.num_antennas, steer_az)
#         g = self.get_ula_gain(self.num_antennas, weights, self.az)
#         self.line.set_ydata(np.abs(g))
#         return self.line,
#
#     # A generator for angles
#     def yield_angle(self):
#         while self.queue.empty():
#             continue
#         # self.steer_az = self.conn.recv()
#         steer_az = iter(self.queue.get().values())
#         self.steer_az = next(iter(self.queue.get().values()))
#         print(self.steer_az)
#         yield self.steer_az
