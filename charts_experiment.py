import matplotlib.pylab as plt
import matplotlib

matplotlib.use('TkAgg')
# plt.ion()
from matplotlib.gridspec import GridSpec
import numpy as np
from time import sleep

print('Using', plt.get_backend(), 'as graphics backend.')
print('Is interactive =', plt.isinteractive())


def make_figure1(param):
    # Data for plotting
    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)
    y1 = np.cos(2 * np.pi * x1 / (param + 1) * np.exp(-x1))
    y2 = np.cos(2 * np.pi * x2 / (param + 1))

    fig = plt.figure(1)
    fig.clear()
    gs = fig.add_gridspec(2)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x1, y1, 'ko-')
    ax1.set(title='A tale of 2 subplots param={}'.format(param), ylabel='Damped oscillation')
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x2, y2, 'r.-')
    ax2.set(xlabel='time (s)', ylabel='Undamped')

    plt.draw()
    plt.pause(.01)


def display_dashboard(param):
    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    fig = plt.figure(num=2, figsize=(18, 9), constrained_layout=True)
    fig.clear()

    gs = GridSpec(nrows=6, ncols=12, figure=fig)

    ax_loss = fig.add_subplot(gs[0:2, 0:3])
    ax_loss.set_title('Loss')
    ax_loss.grid(True, axis='x')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Training Loss')
    ax_loss2 = ax_loss.twinx()
    ax_loss2.set_ylabel('Validation Loss')

    ax_auc = fig.add_subplot(gs[0:2, 3:6])
    ax_auc.set_title('AUC')
    ax_auc.set_xlabel('Epoch')
    ax_auc.set_ylabel('AUC')

    ax_pre = fig.add_subplot(gs[2:4, 0:3])
    ax_pre.set_title('Precision')
    ax_pre.set_xlabel('Epoch')
    ax_pre.set_ylabel('Precision')

    ax_recall = fig.add_subplot(gs[2:4, 3:6])
    ax_recall.set_title('Recall')
    ax_recall.set_xlabel('Epoch')
    ax_recall.set_ylabel('Recall')

    ax_f1 = fig.add_subplot(gs[4:6, 0:3])
    ax_f1.set_title('F1')
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('F1')

    ax_roc = fig.add_subplot(gs[0:3, 6:9])
    ax_roc.set_title('ROC')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')

    ax_prt = fig.add_subplot(gs[0:3, 9:12])
    ax_prt.set_title('Precision and Recall to Threshold')
    ax_prt.set_xlabel('Threshold')
    ax_prt.set_ylabel('Precision')
    ax_prt2 = ax_prt.twinx()
    ax_prt2.set_ylabel('Recall')

    ax_pr = fig.add_subplot(gs[3:6, 6:9])
    ax_pr.set_title('Precision-Recall')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')

    ax_f1t = fig.add_subplot(gs[3:6, 9:12])
    ax_f1t.set_title('F1 to Threshold')
    ax_f1t.set_xlabel('Threshold')
    ax_f1t.set_ylabel('F1')

    fig.suptitle("Dashboard -Epoch {}".format(param))
    # format_axes(fig)

    for ax in fig.get_axes():
        if ax.get_title() in ('Loss', ''):
            continue
        ax.grid(True)
        ax.legend()

    plt.draw()
    plt.pause(.01)


def delay(n=100000000):
    l = ''
    for _ in range(n):
        l = l + 'x'


for i in range(10):
    make_figure1(i)
    display_dashboard(i)
    delay()
    # sleep(3)
