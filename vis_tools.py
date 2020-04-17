import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import numpy as np
import pdb
import plotly
import visdom

from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt


plt.switch_backend("agg")


class visualizer(object):
    def __init__(self, port=8000, scatter_size=[[-1, 1], [-1, 1]], env_name="main"):
        self.env = env_name
        self.vis = visdom.Visdom(port=port)
        (self.x_min, self.x_max), (self.y_min, self.y_max) = scatter_size
        self.counter = 0
        self.plots = {}

    def img_result(self, img_list, caption="view", win=1):
        self.vis.images(
            img_list, nrow=len(img_list), win=win, opts={"caption": caption}
        )

    def plot_img_255(self, img, caption="view", win=1):
        self.vis.image(img, win=win, opts={"caption": caption})

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel="Epochs",
                    ylabel=var_name,
                ),
            )
        else:
            self.vis.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update="append",
            )

    def plot_quiver_img(self, img, flow, win=0, caption="view"):
        fig, ax = plt.subplots(1)
        ax.axis("off")
        ax.imshow(img.transpose(1, 2, 0))
        X, Y, U, V = flow_to_XYUV(flow)
        ax.quiver(X, Y, U, V, angles="xy", color="y")
        plotly_fig = mpl_to_plotly(fig)
        self.vis.plotlyplot(plotly_fig)
        plt.clf()


"""
if __name__ == "__main__":
	print("Main")
"""
