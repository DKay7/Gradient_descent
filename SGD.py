import numpy as np
import matplotlib.pyplot as plt
import plot
import configurations


class StochasticGradientDecent:
    """
    Класс для градиентной оптимимизации функции двух переменных
    """

    def __init__(self):
        self.func = configurations.config['function']
        self.epsilon = configurations.config['epsilon']
        self.alpha = configurations.config['alpha']
        self.max_loop = configurations.config['max_loop']
        self.accuracy = configurations.config['accuracy']
        self.dots_count = configurations.config['dots_count']
        self.save_frequency = configurations.config['save_frequency']

    @staticmethod
    def num_derivative(func, epsilon):
        """
        Функция для приближённого вычисления градиента функции двух переменных.
        :param function func: numpy.ndarray  -> float — произвольная дифференцируемая функция
        :param float epsilon: максимальная величина приращения по осям
        :return: другая функция, которая приближённо вычисляет градиент в точке
        """

        def grad_func(x):
            """
            :param x: np.ndarray — точка, в которой нужно вычислить производную
            :return: градиент np.ndarray
            """
            grd = np.array([], float)
            x_e = np.copy(x)

            for i in range(len(x)):
                x_e[i] += epsilon

                der_v = (func(x_e) - func(x)) / epsilon
                grd = np.append(grd, der_v)

                x_e[i] -= epsilon

            return grd

        return grad_func

    @staticmethod
    def ev_metric(x):
        """
        Функция, считающая длину вектора в эвклидовой метрике.
        Т.е. как корень из суммы квадротов координат.

        :param numpy.ndarray x: вектор

        :return: длина вектора
        """
        sum_sq = 0
        for i in range(len(x)):
            sum_sq += x[i]**2
        return sum_sq**0.5

    def grad_descent_2d(self, low, high, conf=configurations.config, callback=None):
        """
        Реализация градиентного спуска для функций двух переменных
        с несколькими локальным минимумами, но известной квадратной окрестностью
        глобального минимума.

        :param dict conf: словарь параметров градиентного спуска
        :param float low: левая граница интервала по каждой из осей
        :param float high: правая граница интервала по каждой из осей
        :param function callback: функция для отрисовки точек
        :return: координата точки минимума в виде (X,Y)
        """
        func = conf['function']
        dots_count = conf['dots_count']
        alpha = conf['alpha']
        accuracy = conf['accuracy']
        max_loop = conf['max_loop']
        epsilon = conf['epsilon']
        save_frequency = conf['save_frequency']

        grad = self.num_derivative(func, epsilon)

        x_estimate = []
        dots_x = np.linspace(low-1, high+1, dots_count)
        dots_y = np.linspace(low-1, high+1, dots_count)

        dots = np.concatenate([[dots_x, dots_y]]).transpose()
        functions = np.array([0 for _ in range(dots.shape[0])], float)
        iterator = 0

        for iterator in range(max_loop):
            tmp_func = []

            for j in range(dots.shape[0]):

                functions[j] = func(dots[j])
                dots[j] -= alpha * grad(dots[j])

                tmp_func.append(functions[j])

                if self.ev_metric(dots[j]) < accuracy:

                    break

            if iterator % save_frequency == 0:

                while len(tmp_func) < dots.shape[0]:
                    tmp_func.append(0)

                callback(dots, tmp_func)

        y_estimate = functions.min(axis=0, initial=None)

        for i in range(len(functions)):

            if functions[i] == y_estimate:
                x_estimate = dots[i]
                break

        return 'Минимальное значение {0} получено в точке {1}' \
               '\nВсего проведено {2} итераций'.format(y_estimate, x_estimate, iterator)
