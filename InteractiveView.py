from skimage import io
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.pyplot as plt
import numpy as np


class Interactive3D:
    def __init__(self, stack0, stack1, title="Interactive View") -> None:
        # def Interactive3D(stack0, stack1, title="Interactive View"):
        """Creates an interactive view of the overlay created from the control points and the selected correction algorithm"""
        self.stack0 = stack0
        self.stack1 = stack1
        self.title = title
        self.active = 0
        self.max_r = self.stack0.shape[1]
        self.max_c = self.stack1.shape[2]
        self.max_s = self.stack0.shape[0] - 1
        self.alphas = np.ones(self.stack0.shape[1:3])
        if self.max_s == 0:
            self.max_s = 1
        # Generate the figure
        self.create_figure()
        # Show images
        image = self._create_slice(0, 0, 0, 0)
        self.im0_ax = self.ax.imshow(image, cmap="gray", aspect="auto")
        # Put slider on
        self.create_widgets()
        # Enable update functions
        plt.show()

    def get_limits(self, axis, shape):
        self.max_r = shape[0]
        self.max_c = shape[1]
        if axis == 0:
            self.max_s = self.stack0.shape[0] - 1
        elif axis == 1:
            self.max_s = self.stack0.shape[1] - 1
        elif axis == 2:
            self.max_s = self.stack0.shape[2] - 1

    def create_figure(self):
        try:
            plt.close(self.fig)
        except AttributeError:
            pass
        width = self.max_c / self.max_r
        height = 1
        self.fig = plt.figure(num=1, figsize=(8, 8 * height / width), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(self.title)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.80, top=0.92)

    def create_widgets(self, index=0):
        left = self.ax.get_position().x0
        bot = self.ax.get_position().y0
        width = self.ax.get_position().width
        height = self.ax.get_position().height
        axrow = plt.axes([left - 0.05, bot, 0.05, height])
        axrow.set_xlabel("X split point", fontsize=10)
        axcol = plt.axes([left, bot - 0.05, width, 0.05])
        axcol.set_ylabel("Y split point", fontsize=10)
        axslice = plt.axes([left + width + 0.05, bot, 0.05, height])
        axradio = plt.axes([left + width + 0.11, bot + 0.1, 0.05, height / 2])
        axradio.set_title("Plane", fontsize=10)
        self.row_slider = Slider(
            ax=axrow,
            label="",
            valmin=0,
            valmax=self.max_r,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )
        axrow.add_artist(axrow.yaxis)
        axrow.set_yticks([])
        axrow.set_ylabel("Y split point", fontsize=10, labelpad=0)
        self.col_slider = Slider(
            ax=axcol,
            label="",
            valmin=0,
            valmax=self.max_c,
            valinit=0,
            valstep=1,
            orientation="horizontal",
        )
        axcol.add_artist(axcol.xaxis)
        axcol.set_xticks([])
        axcol.set_xlabel("X split point", fontsize=10, labelpad=0)
        self.slice_slider = Slider(
            ax=axslice,
            label="Slice #",
            valmin=0,
            valmax=self.max_s,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )
        # if index == 0:
        options = ("XY", "XZ", "YZ")
        # elif index == 1:
        #     options = ("y", "z", "x")
        # elif index == 2:
        #     options = ("x", "z", "y")
        self.radio = RadioButtons(axradio, options, active=index)
        self.row_slider.on_changed(self.update_row)
        self.col_slider.on_changed(self.update_col)
        self.slice_slider.on_changed(self.change_image)
        self.radio.on_clicked(self.change_plane)

    # Define update functions
    def update_row(self, val):
        self.active = 0
        split_num = int(np.around(val, 0))
        axis = ["XY", "XZ", "YZ"].index(self.radio.value_selected)
        s = self.slice_slider.val
        image = self._create_slice(s, axis, split_num, self.active)
        self.im0_ax.set_data(image)
        self.fig.canvas.draw_idle()
        # val = int(np.around(val, 0))
        # new_alphas = np.copy(self.alphas)
        # new_alphas[:val, :] = 0
        # self.im0_ax.set(alpha=new_alphas[::-1])
        # self.fig.canvas.draw_idle()

    def update_col(self, val):
        self.active = 1
        split_num = int(np.around(val, 0))
        axis = ["XY", "XZ", "YZ"].index(self.radio.value_selected)
        s = self.slice_slider.val
        image = self._create_slice(s, axis, split_num, self.active)
        self.im0_ax.set_data(image)
        self.im0_ax.axes.figure.canvas.draw()
        self.fig.canvas.draw_idle()

    def change_image(self, val):
        val = int(np.around(val, 0))
        axis = ["XY", "XZ", "YZ"].index(self.radio.value_selected)
        if self.active == 0:
            split_num = self.row_slider.val
        elif self.active == 1:
            split_num = self.col_slider.val
        image = self._create_slice(val, axis, split_num, self.active)
        self.im0_ax.set_data(image)
        # self.im1_ax.set_data(self.stack1[val])
        self.ax.set_title(f"{self.title} (Slice {val})")
        self.im0_ax.axes.figure.canvas.draw()
        # self.im1_ax.axes.figure.canvas.draw()
        self.fig.canvas.draw_idle()

    def change_plane(self, index):
        # Handle input
        index = ["XY", "XZ", "YZ"].index(index)
        image = self._create_slice(0, index, 0, 0)
        # Update limits
        self.get_limits(index, image.shape)
        # Update figure
        self.create_figure()
        # Update image
        self.im0_ax = self.ax.imshow(image, cmap="gray")
        self.ax.set_title(f"{self.title} (Slice 0)")
        # Update widgets
        self.create_widgets(index=index)
        plt.show()

    def _create_slice(self, slice_num, axis, split_num, split_axis):
        if axis == 0:
            im0 = self.stack0[slice_num]
            im1 = self.stack1[slice_num]
        elif axis == 1:
            im0 = self.stack0[:, slice_num]
            im1 = self.stack1[:, slice_num]
        elif axis == 2:
            im0 = self.stack0[:, :, slice_num]
            im1 = self.stack1[:, :, slice_num]
        if im0.shape[0] < im0.shape[1] / 2:
            im0 = np.repeat(im0, np.floor(im0.shape[1] / im0.shape[0]), axis=0)
            im1 = np.repeat(im1, np.floor(im1.shape[1] / im1.shape[0]), axis=0)
        elif im0.shape[1] < im0.shape[0] / 2:
            im0 = np.repeat(im0, np.floor(im0.shape[0] / im0.shape[1]), axis=1)
            im1 = np.repeat(im1, np.floor(im1.shape[0] / im1.shape[1]), axis=1)
        if split_axis == 0:
            split_num = im0.shape[0] - split_num
            image = np.vstack((im0[:split_num], im1[split_num:]))
        elif split_axis == 1:
            image = np.hstack((im0[:, :split_num], im1[:, split_num:]))
        return image


def Interactive2D(im0, im1, title="Interactive View"):
    """Creates an interactive view of the overlay created from the control points and the selected correction algorithm"""
    # Generate the figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    max_r = im0.shape[0]
    max_c = im0.shape[1]
    ax.set_title(title)
    alphas = np.ones(im0.shape)
    # Show images
    ax.imshow(im1, cmap="gray")
    im0_ax = ax.imshow(im0, alpha=alphas, cmap="gray")
    # Put slider on
    plt.subplots_adjust(left=0.15, bottom=0.15)
    left = ax.get_position().x0
    bot = ax.get_position().y0
    width = ax.get_position().width
    height = ax.get_position().height
    axrow = plt.axes([left - 0.15, bot, 0.05, height])
    axcol = plt.axes([left, bot - 0.15, width, 0.05])
    row_slider = Slider(
        ax=axrow,
        label="Y pos",
        valmin=0,
        valmax=max_r,
        valinit=max_r,
        valstep=1,
        orientation="vertical",
    )
    col_slider = Slider(
        ax=axcol,
        label="X pos",
        valmin=0,
        valmax=max_c,
        valinit=0,
        valstep=-1,
        orientation="horizontal",
    )

    # Define update functions
    def update_row(val):
        val = int(np.around(val, 0))
        new_alphas = np.copy(alphas)
        new_alphas[:val, :] = 0
        im0_ax.set(alpha=new_alphas[::-1])
        fig.canvas.draw_idle()

    def update_col(val):
        val = int(np.around(val, 0))
        new_alphas = np.copy(alphas)
        new_alphas[:, :val] = 0
        im0_ax.set(alpha=new_alphas)
        fig.canvas.draw_idle()

    # Enable update functions
    row_slider.on_changed(update_row)
    col_slider.on_changed(update_col)
    plt.show()


class Interactive3D_Alignment:
    def __init__(self, stack0, slice_num, title="Interactive View") -> None:
        # def Interactive3D(stack0, stack1, title="Interactive View"):
        """Creates an interactive view of the overlay created from the control points and the selected correction algorithm"""
        self.stack0 = stack0
        self.title = title
        self.slice_num = slice_num
        self.active = 0
        self.max_r = self.stack0.shape[1]
        self.max_c = self.stack0.shape[2]
        self.max_s = self.stack0.shape[0] - 1
        self.alphas = np.ones(self.stack0.shape[1:3])
        if self.max_s == 0:
            self.max_s = 1
        # Generate the figure
        self.create_figure()
        # Show images
        image = self._create_slice(0, 0, 0, 0)
        self.im0_ax = self.ax.imshow(image, cmap="gray", aspect="auto")
        # Put slider on
        self.create_widgets()
        # Enable update functions
        plt.show()

    def get_limits(self, axis, shape):
        self.max_r = shape[0]
        self.max_c = shape[1]
        if axis == 0:
            self.max_s = self.stack0.shape[0] - 1
        elif axis == 1:
            self.max_s = self.stack0.shape[1] - 1
        elif axis == 2:
            self.max_s = self.stack0.shape[2] - 1

    def create_figure(self):
        try:
            plt.close(self.fig)
        except AttributeError:
            pass
        width = self.max_c / self.max_r
        height = 1
        self.fig = plt.figure(num=1, figsize=(8, 8 * height / width), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(self.title)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, bottom=0.16, right=0.80, top=0.96)

    def create_widgets(self, index=0):
        left = self.ax.get_position().x0
        bot = self.ax.get_position().y0
        width = self.ax.get_position().width
        height = self.ax.get_position().height
        axrow = plt.axes([left - 0.05, bot, 0.05, height])
        axcol = plt.axes([left, bot - 0.05, width, 0.05])
        axslice = plt.axes([left + width + 0.05, bot, 0.05, height])
        axradio = plt.axes([left + width + 0.11, bot + 0.1, 0.05, height / 2])
        axradio.set_title("Plane", fontsize=10)
        self.row_slider = Slider(
            ax=axrow,
            label="",
            valmin=-100,
            valmax=100,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )
        axrow.add_artist(axrow.yaxis)
        axrow.set_yticks([])
        axrow.set_ylabel("Y shift", fontsize=10, labelpad=0)
        self.col_slider = Slider(
            ax=axcol,
            label="",
            valmin=-100,
            valmax=100,
            valinit=0,
            valstep=1,
            orientation="horizontal",
        )
        axcol.add_artist(axcol.xaxis)
        axcol.set_xticks([])
        axcol.set_xlabel("X shift", fontsize=10, labelpad=0)
        self.slice_slider = Slider(
            ax=axslice,
            label="Slice #",
            valmin=0,
            valmax=self.max_s,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )
        # if index == 0:
        options = ("XY", "XZ", "YZ")
        # elif index == 1:
        #     options = ("y", "z", "x")
        # elif index == 2:
        #     options = ("x", "z", "y")
        self.radio = RadioButtons(axradio, options, active=index)
        # self.row_slider.on_changed(self.update_row)
        # self.col_slider.on_changed(self.update_col)
        self.row_slider.on_changed(self.update_move)
        self.col_slider.on_changed(self.update_move)
        self.slice_slider.on_changed(self.change_image)
        self.radio.on_clicked(self.change_plane)

    def update_move(self, *args):
        x = self.row_slider.val
        y = self.col_slider.val
        s = self.slice_slider.val
        index = ["XY", "XZ", "YZ"].index(self.radio.value_selected)
        image = self._create_slice(s, index, x, y)
        self.im0_ax.set_data(image)
        self.fig.canvas.draw_idle()

    def change_image(self, val):
        val = int(np.around(val, 0))
        axis = ["XY", "XZ", "YZ"].index(self.radio.value_selected)
        if self.active == 0:
            split_num = self.row_slider.val
        elif self.active == 1:
            split_num = self.col_slider.val
        image = self._create_slice(val, axis, split_num, self.active)
        self.im0_ax.set_data(image)
        self.ax.set_title(f"{self.title} (Slice {val})")
        self.im0_ax.axes.figure.canvas.draw()
        self.fig.canvas.draw_idle()

    def change_plane(self, index):
        # Handle input
        index = ["XY", "XZ", "YZ"].index(index)
        image = self._create_slice(0, index, 0, 0)
        # Update limits
        self.get_limits(index, image.shape)
        # Update figure
        self.create_figure()
        # Update image
        self.im0_ax = self.ax.imshow(image, cmap="gray")
        self.ax.set_title(f"{self.title} (Slice 0)")
        # Update widgets
        self.create_widgets(index=index)
        plt.show()

    def _create_slice(self, slice_num, axis, xshift, yshift):
        # Shift the slice
        new_stack = self.stack0.copy()

        img = new_stack[: self.slice_num + 1]

        row_pad = (abs(yshift), abs(yshift))
        col_pad = (abs(xshift), abs(xshift))
        pad = ((0, 0), row_pad, col_pad, (0, 0))
        img = np.pad(img, pad, "constant")
        # Shift the image
        img = np.roll(img, (yshift, xshift), axis=(1, 2))
        # Crop the image back to original size
        img = img[
            :,
            row_pad[0] : img.shape[1] - row_pad[1],
            col_pad[0] : img.shape[2] - col_pad[1],
            :,
        ]

        new_stack[: self.slice_num + 1] = img

        # Get the image
        if axis == 0:
            im0 = new_stack[slice_num]
        elif axis == 1:
            im0 = new_stack[:, slice_num]
        elif axis == 2:
            im0 = new_stack[:, :, slice_num]

        if im0.shape[0] < im0.shape[1] / 2:
            im0 = np.repeat(im0, np.floor(im0.shape[1] / im0.shape[0]), axis=0)
        elif im0.shape[1] < im0.shape[0] / 2:
            im0 = np.repeat(im0, np.floor(im0.shape[0] / im0.shape[1]), axis=1)

        return im0


def shift_image(img, shift):
    shift = np.array(shift)
    # Pad the image before rolling
    row_pad = (abs(shift[0]), abs(shift[0]))
    col_pad = (abs(shift[1]), abs(shift[1]))
    pad = (row_pad, col_pad, (0, 0))
    img = np.pad(img, pad, "constant")
    # Shift the image
    img = np.roll(img, shift, axis=(0, 1))
    # Crop the image back to original size
    img = img[
        row_pad[0] : img.shape[0] - row_pad[1],
        col_pad[0] : img.shape[1] - col_pad[1],
        :,
    ]
    return img


if __name__ == "__main__":
    import h5py

    aligned_dream3d_path = "/Users/jameslamb/Documents/research/data/CoNi90-thin/CoNi90-thin_aligned.dream3d"
    h5 = h5py.File(aligned_dream3d_path, "r")
    ipf = h5["DataStructure/ImageGeom/Cell Data/IPFColors_001"][:]

    # img = ipf[:82]
    # row_pad = (0, 0)
    # col_pad = (7, 7)
    # pad = ((0, 0), row_pad, col_pad, (0, 0))
    # img = np.pad(img, pad, "constant")
    # # Shift the image
    # img = np.roll(img, (0, 7), axis=(1, 2))
    # # Crop the image back to original size
    # img = img[
    #     :,
    #     row_pad[0] : img.shape[1] - row_pad[1],
    #     col_pad[0] : img.shape[2] - col_pad[1],
    #     :,
    # ]
    # ipf[:82] = img

    Interactive3D_Alignment(ipf, title="IPF Colors", slice_num=81)
