
import numpy as np
import matplotlib.pyplot as plt

from operator import sub


def draw_arrow(x, y, angle,
               ax_pl,
               obj_H=1., obj_A=1., facecolor='black', edgecolor='black',
               arrow_type=1, angle_local_aspect=True):

    x_loc = np.array([x, y])

    # acquire information on aspect
    ax_aspect = get_aspect(ax_pl)

    # determine which angle, and what the rotation point is
    if arrow_type == 1:
        X = np.array([[0, -0.5], [0, 0.5], [1, 0]])
        X_c = np.array([0, 0])
    elif arrow_type == 2:
        X = np.array([[0., 0.], [-0.5, 0.5], [1., 0], [-0.5, -0.5]])
        X_c = np.array([0, 0])
    elif arrow_type == 3:
        X = np.array([[-.5, -.5], [-.5, .5], [.5, .5], [.5, -.5]])
        X_c = np.array([0, 0])

    # scale arrow to specifications
    H = obj_H
    W = H * ax_aspect / obj_A
    X_dim = X.copy()
    X_dim[:, 1] = X_dim[:, 1] * H
    X_dim[:, 0] = X_dim[:, 0] * W

    # adjust angle to aspect
    dx = np.cos(angle)
    dy = np.sin(angle)
    if angle_local_aspect == True:
        rot_ang = np.arctan2(dy * ax_aspect, dx)
    else:
        rot_ang = np.arctan2(dy, dx)

    # adjust aspect
    X_centered2 = X_dim.copy()
    X_centered2[:, 1] = X_centered2[:, 1] * ax_aspect

    # rotate all individual points
    X_rotated = np.full(np.shape(X), np.nan)
    for j in range(np.shape(X)[0]):
        X_rotated[j, :] = vec_rotation(X_centered2[j, :], rot_ang)

    # scale rotated arrow
    X_rotated2 = X_rotated.copy()
    X_rotated2[:, 1] = X_rotated2[:, 1] / ax_aspect

    # place scaled and rotated arrow
    X_transformed = X_rotated2 + x_loc
    ax_pl.add_patch(plt.Polygon(
        X_transformed, facecolor=facecolor, edgecolor=edgecolor))


def get_aspect(ax_A):
    # Total figure size
    figW, figH = ax_A.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax_A.get_position().bounds

    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    data_ratio = sub(*ax_A.get_ylim()) / sub(*ax_A.get_xlim())

    return disp_ratio / data_ratio


def vec_rotation(vector, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def draw_pendulum(ax_p):

    # set parameters -------------------------------------------
    ax_p.set_xlim((np.array([-.7, .7])))
    ax_p.set_ylim((np.array([-0.4, 1.42])))
    ax_p.set_aspect('equal')
    dot_scale = [50, 600]  # size of pendulum bulb
    window = 1.2  # set window size as multiple of pendulum length

    circle_rad = 0.15
    # draw pendulum
    y0 = np.array([np.pi - 0.2, 0])
    x_pos = np.sin(y0[0])
    y_pos = -np.cos(y0[0])
    circle1 = plt.Circle((x_pos, y_pos), circle_rad, color='k', zorder=-6)
    ax_p.add_artist(circle1)

    ax_p.plot([0, x_pos], [0, y_pos], color='k', lineWidth=3, zorder=-6)

    # annotate rod
    ax_p.annotate(r'L', xy=(x_pos / 2 + 0.05, y_pos / 2),
                  fontsize=16, color='k')

    # annotate circle
    ax_p.annotate(r'm', xy=(x_pos + 0.2, y_pos),
                  fontsize=16, color='k')

    # draw theta arrow
    thet = np.linspace(np.pi * 0, np.pi - 0.7, 101)
    r = 0.3
    x = np.sin(thet) * r
    y = -np.cos(thet) * r
    ax_p.plot(x, y, 'k')
    draw_arrow(x[-1], y[-1], thet[-1], ax_p, 0.1)
    ax_p.plot([0, 0], [-0.35, -0.25], 'k')
    ax_p.annotate(r'$\theta$', xy=(0.3, -0.3), fontsize=16)

    # draw theta arrow --------------------------------------------
    thet = np.linspace(np.pi * 0.5, np.pi * 1.7, 101)
    r = 0.1
    x = np.sin(thet) * r
    y = -np.cos(thet) * r
    ax_p.plot(x, y, 'r')
    draw_arrow(x[-1], y[-1], thet[-1], ax_p, 0.1, facecolor='r', edgecolor='r')
    # ax_p.plot( [0,0], [-0.35,-0.25],'r')
#     ax_p.annotate(r'$\tau$' , xy=( -0.2,0),fontsize= 16,color='r')
    ax_p.annotate(r'u', xy=(-0.2, 0.1), fontsize=16, color='r')

    # axis adjust-------------------------------------------
    ax_p.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,      # ticks along the left edge are off
        right=False,         # ticks along the right edge are off
        labelbottom=False,         # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    # draw motion arc
    L = 0.98
    thet = np.linspace(np.pi * 1.2, np.pi * 1.7, 101)
    rad_list = circle_rad + np.array([0.04, 0.08])
    for r in rad_list:
        x = np.sin(thet) * r
        y = -np.cos(thet) * r
        ax_p.plot(x + L * np.sin(y0[0]), y + L *
                  np.cos(y0[1]), 'k', linewidth=0.5)

    # draw horizon
    ax_p.plot([-1, 1], [0, 0], 'k')

    circle1 = plt.Circle((0, 0), 0.03, color='k', zorder=-6)
    ax_p.add_artist(circle1)


if __name__ == '__main__':

    fig_dim = (3, 3)
    fig, ax0 = plt.subplots(1, 1, figsize=(3, 3), dpi=100)
    draw_pendulum(ax0)
    plt.show()
