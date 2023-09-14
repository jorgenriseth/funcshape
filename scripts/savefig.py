from pathlib import Path

from matplotlib import pyplot as plt

FIGPATH = Path(__file__).parent / "../figures/"
def savefig(name, fig=None, suffix=None):
    path = FIGPATH / name
    path.parent.mkdir(exist_ok=True)
    if suffix is not None:
        path = path.with_suffix(suffix)
    if fig is None:
        plt.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path, bbox_inches="tight")
