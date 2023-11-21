from pgo_data.dataset import G2OPGO
from pgo_data.pose_graph import PoseGraph, plot_and_save
import pypose.optim.solver as ppos
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau
import pypose as pp
import torch
import os

DATA_PATH = "./data"
DATA_NAME = "sphere.g2o"
RADIUS = 1e4

if __name__ == "__main__":
    data = G2OPGO(DATA_PATH, DATA_NAME, device='cpu')
    edges, poses, infos = data.edges, data.poses, data.infos
    graph = PoseGraph(data.nodes).to('cpu')
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=RADIUS)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-6, vectorize=False)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

    oriname = os.path.join(DATA_PATH, DATA_NAME.split('.')[0] + '_origin.png')
    axlim = plot_and_save(graph.nodes.translation(), oriname, DATA_NAME)

    scheduler.optimize(input=(edges, poses), weight=infos)

    name = os.path.join(DATA_PATH, DATA_NAME.split('.')[0] + '_' + str(scheduler.steps))
    plot_and_save(graph.nodes.translation(), name + '.png', axlim=axlim)
    torch.save(graph.state_dict(), name+'.pt')