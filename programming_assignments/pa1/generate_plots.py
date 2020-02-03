import matplotlib.pyplot as plt
import os

def generate_plot(title, x_label, y_label, y_vals):
    plt.plot(y_vals)
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
    y_vals = [float(x) for x in input('Enter y-values separated by any whitespace: ').split()]

    generate_plot(title, x_label, y_label, y_vals)