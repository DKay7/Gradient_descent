import plot
import SGD
import configurations
import numpy as np

dots_x = np.linspace(-10, 10, 10)
dots_y = np.linspace(-30, 30, 10)

dots = np.concatenate([[dots_x, dots_y]]).transpose()

plot = plot.Plot()
sdg = SGD.StochasticGradientDecent()
print(sdg.grad_descent_2d(configurations.config['low'],
                          configurations.config['high'],
                          callback=plot.add_dots))
plot.plot_function()
plot.create_animation()
plot.show_animation()
# plot.save_animation('multi_dots_diff_fig_gcd')
