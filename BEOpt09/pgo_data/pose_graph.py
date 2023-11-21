import torch
import pypose as pp
from torch import nn
import matplotlib.pyplot as plt

class PoseGraph(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, edges, poses):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        return error.Log().tensor()


@torch.no_grad()
def plot_and_save(points, pngname, title='', axlim=None):
    points = points.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points[:,0], points[:,1], points[:,2], 'b')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    plt.savefig(pngname)
    print('Saving to', pngname)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()