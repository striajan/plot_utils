import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import warnings


# 2D plotting


def point((x, y), *args, **kwargs):
    """Plot the specified 2D point.

    :type x: int | float
    :type y: int | float
    :rtype: list[matplotlib.lines.Line2D]
    """
    return plt.plot(x, y, *args, **kwargs)


def line((x1, y1), (x2, y2), *args, **kwargs):
    """Plot 2D line specified by its endpoints.

    :type x1: int | float
    :type y1: int | float
    :type x2: int | float
    :type y2: int | float
    :rtype: list[matplotlib.lines.Line2D]
    """
    return plt.plot((x1, x2), (y1, y2), *args, **kwargs)


def line_eq(a, b, c, *args, **kwargs):
    """Plot 2D line specified by the equation a*x + b*y + c = 0. The plotting
    range is determined automatically from the current x and y limits.

    :type a: int | float
    :type b: int | float
    :type c: int | float
    :rtype: list[matplotlib.lines.Line2D]
    """

    # save the original limits
    xlim = plt.xlim()
    ylim = plt.ylim()

    if b != 0:
        x1, x2 = xlim
        y1 = -(a * x1 + c) / float(b)
        y2 = -(a * x2 + c) / float(b)
    elif a != 0:
        y1, y2 = ylim
        x1 = -(b * y1 + c) / float(a)
        x2 = -(b * y2 + c) / float(a)
    else:
        raise ValueError('Either a or b needs to be non-zero.')

    ln = line((x1, y1), (x2, y2), *args, **kwargs)

    # reset the original limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    return ln


def hline(y, *args, **kwargs):
    """Plot horizontal line spanning the current area of the plot.

    :type y: int | float
    :rtype: list[matplotlib.lines.Line2D]
    """
    return line_eq(1, 0, -y, *args, **kwargs)


def vline(x, *args, **kwargs):
    """Plot vertical line spanning the current area of the plot.

    :type x: int | float
    :rtype: list[matplotlib.lines.Line2D]
    """
    return line_eq(1, 0, -x, *args, **kwargs)


def polyline(pts, *args, **kwargs):
    """Plot polyline specified by its points.

    :type pts: numpy.ndarray | list
    :rtype: list[matplotlib.lines.Line2D]
    """
    pts = np.asarray(pts)
    return plt.plot(pts[:,0], pts[:,1], *args, **kwargs)


def polygon(pts, *args, **kwargs):
    """Plot 2D points specified by their coordinates x and y in a closed loop.

    :type pts: numpy.ndarray | list
    :rtype: list[matplotlib.lines.Line2D]
    """
    pts = np.asarray(pts)
    if pts.shape[0] > 0:
        pts = np.vstack((pts, pts[0]))
    return plt.plot(pts[:,0], pts[:,1], *args, **kwargs)


def vector(p, phi, len=1, *args, **kwargs):
    """Plot vector from the specified point having the specified orientation and length.

    :type p: numpy.array | tuple
    :type phi: float
    :type len: float
    :rtype: list[matplotlib.lines.Line2D]
    """
    p = np.asarray(p)
    u = np.array((np.cos(phi), np.sin(phi)))
    return line(p, p + len * u, *args, **kwargs)


def orientation(phi, len=1, *args, **kwargs):
    """Plot oriented vector starting in the origin and having the specified length.

    :type phi: float
    :type len: float
    :rtype: list[matplotlib.lines.Line2D]
    """
    return vector((0, 0), phi, len, *args, **kwargs)


def polyvector(pts, *args, **kwargs):
    """Plot oriented polyline with its start marked by circle and end by arrow.

    :type pts: numpy.array | list
    :rtype: list[matplotlib.lines.Line2D]
    """
    line = polyline(pts, *args, **kwargs)
    p1 = point(pts[0], *args, marker='o', **kwargs)
    dx = pts[-1][0] - pts[-2][0]
    dy = pts[-1][1] - pts[-2][1]
    if plt.gca().yaxis_inverted():
        dy = -dy
    angle = np.arctan2(dy, dx)
    angle = 180 * angle / np.pi - 90
    p2 = point(pts[-1], *args, marker=(3, 0, angle), **kwargs)
    return line + p1 + p2


def circle((x, y), radius, **kwargs):
    """Plot circle with the specified center (x, y) and radius.

    :type x: int | float
    :type y: int | float
    :type radius: int | float
    :rtype: matplotlib.patches.Circle
    """
    circle = plt.Circle((x, y), radius=radius, **kwargs)
    ax = plt.gcf().gca()
    ax.add_artist(circle)
    return circle


def rectangle(rect, *args, **kwargs):
    """Plot rectangle specified as usual in OpenCV, i.e. by its center point,
    width and height, and orientation angle in degrees.

    :param rect: ((center_x, center_y), (width, height), rot_deg)
    :type rect: ((float, float), (float, float), float)
    :rtype: list[matplotlib.lines.Line2D]
    """
    p = np.asarray(rect[0])
    w, h = rect[1]
    ang = np.deg2rad(rect[2])
    cos, sin = np.cos(ang), np.sin(ang)
    u = 0.5 * w * np.array((cos, sin))
    v = 0.5 * h * np.array((-sin, cos))
    pts = np.array((p + u + v, p - u + v, p - u - v, p + u - v))
    return polygon(pts, *args, **kwargs)


def box((x1, y1), (x2, y2), *args, **kwargs):
    """Plot axes-aligned box specified by its top-left corner (x1,y1) and its
    bottom-right corner (x2, y2).

    :rtype: list[matplotlib.lines.Line2D]
    """
    pts = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    return polygon(pts, *args, **kwargs)


# 3D plotting


def scatter(x, y, z, *args, **kwargs):
    """Plot 3D points specified by (x,y,z) coordinates.

    :type x: numpy.array
    :type y: numpy.array
    :type z: numpy.array
    :rtype: mpl_toolkits.mplot3d.art3d.Patch3DCollection
    """
    ax = plt.gcf().gca(projection='3d')
    return ax.scatter(x, y, z, *args, **kwargs)


def mesh(x, y, z, *args, **kwargs):
    """Plot 3D mesh specified by (x,y,z) coordinates.

    :type x: numpy.array
    :type y: numpy.array
    :type z: numpy.array
    :rtype: mpl_toolkits.mplot3d.art3d.Line3DCollection
    """
    ax = plt.gcf().gca(projection='3d')
    return ax.plot_wireframe(x, y, z, *args, rstride=1, cstride=1, **kwargs)


def surf(x, y, z, *args, **kwargs):
    """Plot 3D surface specified by (x,y,z) coordinates.

    :type x: numpy.array
    :type y: numpy.array
    :type z: numpy.array
    :rtype: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    ax = plt.gcf().gca(projection='3d')
    return ax.plot_surface(x, y, z, *args, rstride=1, cstride=1, **kwargs)


def surfz(z, *args, **kwargs):
    """Plot 3D surface specified by (x,y,z) coordinates where (x,y)
    coordinates are obtained automatically from shape of z.

    :type z: numpy.array
    :rtype: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    x = range(z.shape[1])
    y = range(z.shape[0])
    x, y = np.meshgrid(x, y)
    return surf(x, y, z, *args, **kwargs)


def create_ellipsoid(mean, cov, num_theta, num_phi, **kwargs):
    """Create ellipsoid specified by its mean and covariance matrix.
    The elipsoid is discretized along spherical angles as specified.
    Return [x, y, z] coordinates of the vertices.

    :type mean: numpy.ndarray
    :type cov: numpy.ndarray
    :type num_theta: int
    :type num_theta: int
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    # add one more theta so that the ellipse is closed loop
    num_theta += 1

    # find the rotation matrix and scales of the axes
    _, s, v = np.linalg.svd(cov)
    s = np.sqrt(s)

    # generate spherical angles
    theta = np.linspace(0, 2 * np.pi, num_theta)
    phi = np.linspace(0, np.pi, num_phi)

    # generate 3D coordinates of points lying on the sphere whose
    # radius is 1 and its center is in [0,0,0]
    x = np.outer(np.cos(theta), np.sin(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.ones(num_theta), np.cos(phi))

    # transform the point by scaling, rotation and translation
    for i in range(num_theta):
        for j in range(num_phi):
            # scale along the axes according to the covariance
            w = np.multiply(s, (x[i,j], y[i,j], z[i,j]))
            # rotate according to the covariance
            w = np.dot(w, v)
            # translate by the mean
            x[i,j], y[i,j], z[i,j] = mean + w

    return x, y, z


def ellipsoid_wireframe(mean, cov, num_theta=16, num_phi=16, **kwargs):
    """Plot wireframe model of the ellipsoid specified by its mean and covariance matrix.

    :type mean: numpy.ndarray
    :type cov: numpy.ndarray
    :type num_theta: int
    :type num_theta: int
    :rtype: mpl_toolkits.mplot3d.art3d.Line3DCollection
    """
    x, y, z = create_ellipsoid(mean, cov, num_theta, num_phi)
    return mesh(x, y, z, **kwargs)


def ellipsoid_surface(mean, cov, num_theta=16, num_phi=16, **kwargs):
    """Create 3D ellipsoid specified by its mean and covariance matrix.

    :type mean: numpy.ndarray
    :type cov: numpy.ndarray
    :type num_theta: int
    :type num_theta: int
    :rtype: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    x, y, z = create_ellipsoid(mean, cov, num_theta, num_phi)
    return surf(x, y, z, **kwargs)


def rgb(rgb, max_plot=0, **kwargs):
    """Plot maximally specified number of RGB points in their color.

    :type rgb: numpy.ndarray
    :type max_plot: int
    :rtype: mpl_toolkits.mplot3d.art3d.Patch3DCollection
    """

    # if there are too many points to be plotted, sample them uniformly
    if 0 < max_plot < rgb.shape[0]:
        rgb_ind = np.asarray(np.linspace(0, rgb.shape[0] - 1, max_plot), int)
        rgb = rgb[rgb_ind]

    # scatter the points
    return scatter(rgb[:,0], rgb[:,1], rgb[:,2], c=rgb, **kwargs)


# plotting start and finish


def figure(win_name='Plot'):
    """Start plotting to a window having the specified name.

    :type win_name: str
    :rtype: matplotlib.figure.Figure
    """
    return plt.figure(win_name)


def show(block=True, equal_aspect=False, border_width=0, show_grid=False, close_esc=True):
    """Show the plot with possible additional settings.

    :type block: bool
    :type equal_aspect: bool
    :type border_width: float
    :type show_grid: bool
    :type close_esc: bool
    """
    if equal_aspect:
        ax = plt.gca()
        ax.set_aspect('equal')
    if border_width > 0:
        set_border(border_width)
    if show_grid:
        plt.grid(show_grid)
    if close_esc:
        fig = plt.gcf()
        def on_key_pressed(event):
            if event.key == 'escape':
                plt.close(fig)
        fig.canvas.mpl_connect('key_press_event', on_key_pressed)
    if not plt.isinteractive():
        plt.show(block)


# images


def image(img, show_axes=True, show_colorbar=True, gray=False, **kwargs):
    """Plot the specified image using PyPlot. Specify visibility of axes and color-bar,
    and whether the image is grayscale.

    :type img: numpy.ndarray
    :type show_axes: bool
    :type show_colorbar: bool
    :type gray: bool
    :rtype: matplotlib.image.AxesImage
    """
    im = plt.imshow(img, interpolation='none', **kwargs)
    if gray:
        plt.set_cmap('gray')
    if not show_axes:
        axes_visibility(False)
    if show_colorbar and (img.ndim == 2) and (not gray):
        plt.colorbar()
    return im


def imshow(img, win_name='Image', show_axes=True, show_colorbar=True, block=True, gray=False, **kwargs):
    """Plot the specified image using PyPlot and show it. Specify window name, visibility
    of axes and color-bar, blockage of the window, and whether the image is grayscale.

    :type img: numpy.ndarray
    :type win_name: str
    :type show_axes: bool
    :type show_colorbar: bool
    :type block: bool
    :type gray: bool
    :rtype: matplotlib.figure.Figure
    """
    fig = figure(win_name)
    image(img, show_axes, show_colorbar, gray, **kwargs)
    show(block)
    return fig


def imarray(imgs, win_name='Image array', img_names=None, show_axes=True, show_colorbar=True,
            block=True, gray=False, screen_aspect=1.5, item_aspect=1.0, fig=None, **kwargs):
    """Plot the specified array of images using PyPlot and show it. Specify whether color-bar
    should be shown (and shared among all images).

    :type imgs: list[numpy.ndarray]
    :type win_name: str
    :type img_names: list[str]
    :type show_axes: bool
    :type show_colorbar: bool
    :type block: bool
    :type gray: bool
    :type screen_aspect: float
    :type item_aspect: float
    :type fig: matplotlib.figure.Figure
    :rtype: matplotlib.figure.Figure
    """

    if fig is None:
        fig = figure(win_name)

    num_imgs = len(imgs)

    # range of image values needed for common colorbar
    if show_colorbar:
        if not kwargs.has_key('vmin'):
            kwargs['vmin'] = min([np.min(img) for img in imgs])
        if not kwargs.has_key('vmax'):
            kwargs['vmax'] = max([np.max(img) for img in imgs])

    # show properly laid-out images
    num_rows, num_cols = layout(num_imgs, screen_aspect, item_aspect)
    for i in range(num_imgs):
        plt.subplot(num_rows, num_cols, i + 1)
        if (img_names is not None) and (i < len(img_names)):
            plt.title(img_names[i])
        img_ax = image(imgs[i], show_axes, False, gray, **kwargs)

    # show common color-bar
    if show_colorbar and all([img.ndim == 2 for img in imgs]) and (not gray):
        fig = plt.gcf()
        fig.subplots_adjust(right=0.8)
        ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
        fig.colorbar(img_ax, cax=ax)

    show(block)
    return fig


def imstitch(imgs, margin=0, num_rows=-1, num_cols=-1):
    """Stitch multiple images to single image.

    :type imgs: list[numpy.ndarray]
    :type margin: int
    :type num_rows: int
    :type num_cols: int
    :rtype: numpy.ndarray
    """

    num_imgs = len(imgs)
    h, w, c = imgs[0].shape

    if num_rows == -1 and num_cols == -1:
        img_aspect = w / float(h)
        num_rows, num_cols = layout(num_imgs, item_aspect=img_aspect)
    elif num_rows == -1:
        num_rows = int(np.ceil(num_imgs / float(num_cols)))
    elif num_cols == -1:
        num_cols = int(np.ceil(num_imgs / float(num_rows)))

    h_stitch = num_rows * h + (num_rows - 1) * margin
    w_stitch = num_cols * w + (num_cols - 1) * margin
    img_stitch = np.zeros([h_stitch, w_stitch, c], dtype=imgs[0].dtype)

    for i in range(num_imgs):
        r = (i / num_cols) * (h + margin)
        c = (i % num_cols) * (w + margin)
        img_stitch[r:r+h,c:c+w] = imgs[i]

    return img_stitch


def image_poi(img, win_name='ROI definition'):
    """Define and return points of interest (POI) in the image using PyPlot.

    :type img: numpy.ndarray
    :type win_name: str
    :rtype: numpy.ndarray
    """
    plt.figure(win_name)
    plt.imshow(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poi = plt.ginput(n=np.iinfo(int).max, timeout=0, show_clicks=True)
    plt.close(win_name)
    return np.array(poi)


# figure settings


def grid(x, y, show_ticks_labels=True, color='k', linestyle='-'):
    """Show grid specified by list of x and y coordinates.

    :type x: numpy.ndarray
    :type y: numpy.ndarray
    """
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.grid(which='major', color=color, linestyle=linestyle)
    if not show_ticks_labels:
        hide_ticks_labels()


def axes_visibility(visible=True):
    """Show or hide axes of the current plot.

    :type visible: bool
    """
    ax = plt.gca()
    ax.axis('on' if visible else 'off')


def hide_ticks_labels():
    """Hide all x and y ticks labels."""
    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])


def fit_image_border(img):
    """Fit borders of the plot to the specified image.

    :type img: numpy.ndarray
    """
    h, w = img.shape[0:2]
    plt.xlim(-0.5, w - 0.5)
    plt.ylim(h - 0.5, -0.5)


def set_border(rel_border=0.1):
    """Add border to axis limits of the current plot. The border size
    is set relatively to the current range of limits.

    :type rel_border: float
    :rtype: list[int | float]
    """
    [xmin, xmax, ymin, ymax] = plt.axis()
    rng = max([xmax - xmin, ymax - ymin])
    border = rel_border * rng
    return plt.axis([xmin - border, xmax + border, ymin - border, ymax + border])


def move_window(x=0, y=0):
    """Move PyPlot window to the specified location if possible.

    :type x: int
    :type y: int
    """
    try:
        manager = plt.get_current_fig_manager()
        if manager is not None:
            # each backend has to be treated individually
            backend = plt.get_backend()
            window = manager.window
            if backend == 'Qt4Agg':
                window.move(x, y)
            elif backend == 'TkAgg':
                window.wm_geometry('+{0}+{1}'.format(x, y))
            elif backend in ['GTKAgg', 'GTK3Agg', 'GTK', 'GTKCairo', 'GTK3Cairo']:
                window.set_position((x, y))
            elif backend in ['WXAgg', 'WX']:
                window.SetPosition((x, y))
    except:
        pass


def layout(num_item, screen_aspect=1.5, item_aspect=1.0):
    """Compute optimum number of rows and columns for the specified number of items
    having the specified aspect to be laid on the screen with the specified aspect.

    :type num_item: int
    :type screen_aspect: float
    :type item_aspect: float
    :rtype: (int, int)
    """

    aspect = screen_aspect / item_aspect

    cols1 = np.ceil(np.sqrt(aspect * num_item))
    rows1 = np.ceil(num_item / cols1)

    rows2 = np.ceil(np.sqrt(num_item / aspect))
    cols2 = np.ceil(num_item / rows2)

    if (rows1 * cols1) <= (rows2 * cols2):
        rows, cols = rows1, cols1
    else:
        rows, cols = rows2, cols2

    return int(rows), int(cols)


def save_axis(file_path):
    """Save the current axis to the specified file.

    :type file_path: str
    """
    fig = plt.gcf()
    ax = plt.gca()
    ext = ax.get_window_extent()
    ext = ext.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(file_path, bbox_inches=ext)
