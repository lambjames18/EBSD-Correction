
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np    

def Interactive3D(stack0, stack1, title="Interactive View"):
    """Creates an interactive view of the overlay created from the control points and the selected correction algorithm"""
    # Generate the figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    max_r = stack0.shape[1]
    max_c = stack1.shape[2]
    max_s = stack0.shape[0] - 1
    if max_s == 0: max_s = 1
    ax.set_title(title)
    alphas = np.ones(stack0.shape[1:3])
    # Show images
    im1_ax = ax.imshow(stack1[0], cmap="gray")
    im0_ax = ax.imshow(stack0[0], alpha=alphas, cmap="gray")
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
        valinit=max_c,
        valstep=1,
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

    def change_image(val):
        val = int(np.around(val, 0))
        im0_ax.set_data(stack0[val])
        im1_ax.set_data(stack1[val])
        ax.set_title(f"{title} (Slice {val})")
        im0_ax.axes.figure.canvas.draw()
        im1_ax.axes.figure.canvas.draw()
        fig.canvas.draw_idle()

    # Enable update functions
    row_slider.on_changed(update_row)
    col_slider.on_changed(update_col)
    # Create slice slider if need be
    axslice = plt.axes([left + 0.65, bot, 0.05, height])
    slice_slider = Slider(
        ax=axslice,
        label="Slice #",
        valmin=0,
        valmax=max_s,
        valinit=0,
        valstep=1,
        orientation="vertical",
    )
    slice_slider.on_changed(change_image)
    plt.show()

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
    im1_ax = ax.imshow(im1, cmap="gray")
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
        valinit=max_c,
        valstep=1,
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
    im0 = np.random.randint(0, 255, (100, 100))
    im1 = np.random.randint(0, 255, (100, 100))
    Interactive2D(im0, im1)

    stack0 = np.random.randint(0, 255, (100, 100, 100))
    stack1 = np.random.randint(0, 255, (100, 100, 100))
    Interactive3D(stack0, stack1)