
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np
import numpy.random as random

import configurations

# TODO сделать вид сверху


class Plot:
    """
    Класс для построения графика функции и отрисовки анимации градиентного спуска
    """
    def __init__(self):

        self.frames = None
        self.fps = configurations.config['fps']
        self.func = configurations.config['function']
        self.low = configurations.config['low']
        self.high = configurations.config['high']
        self.dots_count = configurations.config['dots_count']
        self.plot_alpha = configurations.config['plot_alpha']
        self.c_map = configurations.config['c_map']
        self.markers = configurations.config['markers']
        self.line_style = configurations.config['line_style']
        self.marker_size = configurations.config['marker_size']
        self.shown_dots_start = configurations.config['shown_dots_start']
        self.shown_dots_start = configurations.config['shown_dots_start']
        self.shown_dots_end = configurations.config['shown_dots_end']

        self.plt = matplotlib.pyplot
        self.fig = self.plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(self.low, self.high)
        self.ax.set_ylim(self.low, self.high)
        self.ax.set_zlim(-1, 0.75)
        self.sct = [self.ax.plot([], [], [],
                                 linestyle=self.line_style,
                                 marker=self.markers[random.randint(0, len(self.markers))],
                                 markersize=self.marker_size,
                                 )[0]
                    for _ in range(self.dots_count)]

        self.dots = np.array([], float).reshape((0, self.dots_count, 2))
        self.dots_x = []
        self.dots_y = []
        self.functions = np.array([], float).reshape((0, self.dots_count))

        self.ani = None

    def update(self, iterator, dots, functions):
        """
        Функция для обновления точек на графике. Используется в анимации градиентного спуска.
        Добавляет на график все точки от 0 до i, где i текущий кадр.
        :param int iterator: номер текущего кадра. Подается функцией анимации
        :param list dots: массив X-, Y- координат точек
        :param list functions: массив Z-координат точек
        """

        for i in range(self.shown_dots_start, self.shown_dots_end):
            self.sct[i].set_data(dots[:iterator, i, 0], dots[:iterator, i, 1])
            self.sct[i].set_3d_properties(functions[:iterator, i])

    def plot_function(self, key=None):
        """
        Функция (процедера) отрисовки графика функции, для которой выполняется оптимизация.
        :return:
        """
        lim_x = [np.min(self.dots[:, :, 0]), np.max(self.dots[:, :, 0])]
        lim_y = [np.min(self.dots[:, :, 1]), np.max(self.dots[:, :, 1])]
        lim_z = [np.min(self.functions[:]), np.max(self.functions[:])]

        self.ax.set_xlim(*lim_x)
        self.ax.set_ylim(*lim_y)
        self.ax.set_zlim(*lim_z)

        self.ax.legend((self.sct[i]
                        for i in range(self.shown_dots_start, self.shown_dots_end)),
                       ['Точка номер {0}'.format(i)
                        for i in range(self.shown_dots_start, self.shown_dots_end)])

        self.ax.set_xlabel('X', fontsize=15)
        self.ax.set_ylabel('Y', fontsize=15)
        self.ax.set_zlabel('Z', fontsize=15)

        real_dots_cnt = self.shown_dots_end-self.shown_dots_start

        if real_dots_cnt % 10 == 1:
            end_string = 'ка'
        elif real_dots_cnt % 10 <= 4:
            end_string = 'ки'
        else:
            end_string = 'ек'

        self.ax.set_title('Стохастический градиентный спуск с {0} точками\n показан{1} {2} точ{3}'
                          .format(self.dots_count,
                                  ('ы' if (self.shown_dots_end-self.shown_dots_start) > 1 else 'а'),
                                  (self.shown_dots_end - self.shown_dots_start),
                                  end_string)
                          )

        x = np.linspace(self.low if key == 'hide' else lim_x[0],
                        self.high if key == 'hide' else lim_x[1], 100)

        y = np.linspace(self.low if key == 'hide' else lim_y[0],
                        self.high if key == 'hide' else lim_y[1], 100)
        x, y = np.meshgrid(x, y)
        z = self.func([x, y])

        self.ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=self.c_map, alpha=self.plot_alpha)

    def add_dots(self, dot, function):
        """
        Callback-функция, вызываемая на каждом шаге градиентного спуска.
        Добавляет координаты расчитанной точки и функцию в этой точке в соответствиующие массивы

        :param numpy.ndarray dot: координаты очередной точки
        :param float function: значение функции в этой точке
        """
        self.dots = np.append(self.dots, [dot], axis=0)
        self.functions = np.append(self.functions, [function], axis=0)

    def create_animation(self):
        """
        Функция, расчитывающая анимацию построения и связи точек градиентного спуска
        """
        self.functions = np.array(self.functions, float)
        self.frames = self.dots.shape[0]
        self.ani = animation.FuncAnimation(self.fig,
                                           self.update,
                                           self.frames,
                                           fargs=(self.dots, self.functions),
                                           interval=1000/self.fps,
                                           repeat=True)

    def show_animation(self):
        """
        Функция, запускающая показ анимации
        """
        self.plt.show()

    def save_animation(self, name):
        """
        Функция, сохраняющая анимацию в .gif файл.

        :type name: str
        :param name: Имя файла.
        """
        self.ani.save('Animated_SGD\\'+name+'.gif', writer='imagemagick', fps=self.fps)
