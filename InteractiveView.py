
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
        if self.max_s == 0: self.max_s = 1
        # Generate the figure
        self.create_figure()
        # Show images
        image = self._create_slice(0, 0, 0, 0)
        self.im0_ax = self.ax.imshow(image, cmap="gray")
        # Put slider on
        self.create_widgets()
        # Enable update functions
        plt.show()

    def get_limits(self, axis):
        if axis == 0:
            self.max_s = self.stack0.shape[0] - 1
            self.max_r = self.stack0.shape[1]
            self.max_c = self.stack0.shape[2]
        elif axis == 1:
            self.max_s = self.stack0.shape[1] - 1
            self.max_r = self.stack0.shape[0]
            self.max_c = self.stack0.shape[2]
        elif axis == 2:
            self.max_s = self.stack0.shape[2] - 1
            self.max_r = self.stack0.shape[0]
            self.max_c = self.stack0.shape[1]

    def create_figure(self):
        plt.clf()
        width = self.max_c / self.max_r
        height = 1
        self.fig = plt.figure(num=1, figsize=(10, 10 * height / width))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(self.title)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8)

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
        axradio = plt.axes([left + width + 0.12, bot + height / 2, 0.05, 0.1])
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
        # val = int(np.around(val, 0))
        # new_alphas = np.copy(self.alphas)
        # new_alphas[:, :val] = 0
        # self.im0_ax.set(alpha=new_alphas)
        # self.fig.canvas.draw_idle()

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
        # Update limits
        self.get_limits(index)
        # Update figure
        self.create_figure()
        # Update image
        image = self._create_slice(0, index, 0, 0)
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
        valmin=max_c,
        valmax=0,
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


if __name__ == "__main__":
    stack0 = np.random.rand(80, 100, 100)
    stack1 = np.random.rand(80, 100, 100) * 10
    Interactive3D(stack0, stack1)
