import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    print(frame_number)

    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    #
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i,joints in enumerate(mp_joints):

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)


        #     print(data.shape)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        # data[:, :, 0] -= data[0:1, 0:1, 0]
        # data[:, :, 0] += mp_offset[i]
        #
        # data[:, :, 2] -= data[0:1, 0:1, 2]
        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 15#7.5
        #         ax =
        plot_xzPlane(-3, 3, 0, -3, 3)
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(data["joints"][index, chain, 0], data["joints"][index, chain, 1], data["joints"][index, chain, 2], linewidth=linewidth,
                          color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_multi(
        save_path,
        kinematic_tree,
        joints_all,
        title,
        figsize=(10, 10),
        fps=120,
        radius=4
):
    """
    Plot 3D motion of any number of characters and save it as a video.

    Args:
        save_path (str): Path to output video file (with extension, e.g., .mp4).
        kinematic_tree (list of list of int): A list where each sublist contains joint indices forming a bone chain.
        joints_all (np.ndarray): Array of shape (num_characters, num_frames, num_joints, 3),
                                 representing the (x, y, z) coordinates of all characters at each frame.
        title (str): Title of the video (will automatically wrap if longer than 10 words).
        figsize (tuple): Figure size, default is (10, 10).
        fps (int): Frames per second, default is 120.
        radius (float): Controls camera's field of view (related to axis limits), default is 4.
    """

    num_characters = joints_all.shape[0]

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([
            ' '.join(title_sp[:10]),
            ' '.join(title_sp[10:])
        ])

    data_all = [joints_all[p].copy() for p in range(num_characters)]
    num_frames = data_all[0].shape[0]

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    init()

    stacked = np.concatenate(data_all, axis=0)  # 形状 (num_characters * num_frames, num_joints, 3)
    MINS = stacked.reshape(-1, 3).min(axis=0)  # [min_x, min_y, min_z]
    MAXS = stacked.reshape(-1, 3).max(axis=0)  # [max_x, max_y, max_z]

    colors = [
        'red', 'blue', 'black', 'darkred', 'darkblue',
        'darkgreen', 'purple', 'orange', 'brown', 'pink',
        'darkcyan', 'gold', 'magenta', 'olive', 'teal'
    ]
    text_colors = [
        'red', 'blue', 'green', 'darkred', 'darkblue',
        'darkgreen', 'purple', 'orange', 'brown', 'pink'
    ]
    if num_characters > len(text_colors):
        extra = num_characters - len(text_colors)
        text_colors += [text_colors[i % len(text_colors)] for i in range(extra)]

    height_offset = MINS[1]
    for p in range(num_characters):
        data_all[p][:, :, 1] -= height_offset

    trajecs = [data_all[p][:, 0, :] for p in range(num_characters)]

    last_text = []
    for p in range(num_characters):
        x0, y0, z0 = trajecs[p][0, 0], data_all[p][0, 15, 1] + 0.2, trajecs[p][0, 2]
        txt = ax.text(
            x0, y0, z0,
            f'Person {p + 1}',
            color=text_colors[p]
        )
        last_text.append(txt)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    def update(frame_idx):
        nonlocal last_text

        ax.lines = []
        ax.collections = []

        ax.view_init(elev=120, azim=-90)
        ax.dist = 10

        plot_xzPlane(
            MINS[0] - 0.2, MAXS[0] + 0.2,
            0,
            MINS[2] - 0.2, MAXS[2] + 0.2
        )

        for p in range(num_characters):
            last_text[p].remove()
            x_p = trajecs[p][frame_idx, 0]
            y_p = data_all[p][frame_idx, 15, 1] + 0.2
            z_p = trajecs[p][frame_idx, 2]
            last_text[p] = ax.text(x_p, y_p, z_p, f'Person {p + 1}', color=text_colors[p])

        if frame_idx > 1:
            for p in range(num_characters):
                ax.plot3D(
                    trajecs[p][:frame_idx, 0],
                    np.zeros_like(trajecs[p][:frame_idx, 0]),
                    trajecs[p][:frame_idx, 2],
                    linewidth=1.0,
                    color=text_colors[p]
                )

        for chain_idx, chain in enumerate(kinematic_tree):
            linewidth = 4.0 if chain_idx < 5 else 2.0
            c = colors[chain_idx % len(colors)]
            for p in range(num_characters):
                ax.plot3D(
                    data_all[p][frame_idx, chain, 0],
                    data_all[p][frame_idx, chain, 1],
                    data_all[p][frame_idx, chain, 2],
                    linewidth=linewidth,
                    color=c
                )

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        repeat=False
    )

    ani.save(save_path, fps=fps)
    plt.close()