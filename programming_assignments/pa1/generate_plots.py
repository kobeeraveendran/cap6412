import matplotlib.pyplot as plt
import os

def generate_plot(title, x_label, y_label, x_vals, y1_vals, y2_vals = None, annotate = False):
    
    if not y2_vals:
        plt.plot(x_vals, y1_vals, marker = 'o')
    else:
        plt.plot(x_vals, y1_vals, marker = 'o', label = 'train')
        plt.plot(x_vals, y2_vals, marker = '^', label = 'val')
        plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    os.makedirs('plots', exist_ok = True)
    print("Saving plot into 'plots/' ...")

    plt.savefig('plots/{}.png'.format(title.replace(' ', '_')))

if __name__ == '__main__':
    print('Enter plot information...')
    title = input('Title: ')
    x_label = input('X-axis label: ')
    y_label = input('Y-axis label: ')
    #x_vals = [2.13, 6.91, 12.95]
    #y1_vals = [490.99, 367.47, 1002.32]
    #y2_vals = [1.64, 1.51, 1.44, 1.39, 1.35, 0.45, 0.34, 0.31, 0.29, 0.29, 0.27, 0.26, 0.25, 0.25, 0.24]
    x_vals = [float(x) for x in input('Enter x-values separated by any whitespace: ').split()]
    y1_vals = [float(x) for x in input('Enter y-values separated by any whitespace: ').split()]
    #x_label = 'Number of parameters (in millions)'
    #y_label = 'Training time (seconds)'
    generate_plot(title, x_label, y_label, x_vals, y1_vals)