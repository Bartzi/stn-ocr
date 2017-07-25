import argparse
import json

import matplotlib
import re
from collections import defaultdict

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class LogPlotter(object):

    def __init__(self, log_file):
        self.log_file = log_file
        self.iterations_per_epoch = None
        self.train_iterations = {}
        self.test_iterations = {}

    def parse_log_file(self, start=0, end=None):
        last_iteration = 0

        with open(self.log_file) as log_file:
            for line in log_file:

                line_splits = line.split('\t')
                if len(line_splits) == 3:
                    header = line_splits[0]
                    info = line_splits[2]
                elif len(line_splits) == 1:
                    info = line_splits[0]
                    if 'Validation' in info:
                        event_info = re.search(r'.*-(?P<event_name>.*)=(?P<value>.*)', info)
                        event_key = event_info.group('event_name')
                        event_value = event_info.group('value')

                        iteration_info = self.test_iterations.get(last_iteration, {})
                        iteration_info[event_key] = float(event_value)
                        self.test_iterations[last_iteration] = iteration_info
                    elif 'EPOCH SIZE' in info:
                        self.iterations_per_epoch = int(info.split(':')[-1].strip())
                    continue
                else:
                    continue

                iteration_info = re.search(r'Epoch\[(?P<epoch>\d+)\] Batch \[(?P<batch>\d+)\]', header)

                epoch = int(iteration_info.group('epoch'))
                epoch_iteration = int(iteration_info.group('batch'))

                if self.iterations_per_epoch is None:
                    raise ValueError("Iterations per epoch not found in header of log file, can not plot log")
                iteration = epoch * self.iterations_per_epoch + epoch_iteration

                if iteration < start:
                    continue

                last_iteration = iteration
                event_info = re.search(r'.*-(?P<event_name>.*)=(?P<value>.*)', info)
                event_key = event_info.group('event_name')
                event_value = event_info.group('value')

                iteration_info = self.train_iterations.get(iteration, {})
                iteration_info[event_key] = float(event_value)
                self.train_iterations[iteration] = iteration_info

                if end is not None and iteration > end:
                    break

    def smooth_values(self, values, smooth_interval=1000):
        sorted_iterations = list(sorted(values.keys()))
        smoothed_values = {}
        values_to_smooth = []
        for iteration in sorted_iterations:
            infos = values[iteration]
            if iteration % smooth_interval != 0:
                values_to_smooth.append(infos)
                continue

            accumulator = defaultdict(float)
            for value_to_smooth in values_to_smooth:
                for metric, metric_value in value_to_smooth.items():
                    accumulator[metric] += metric_value

            for metric in accumulator.keys():
                accumulator[metric] /= len(values_to_smooth) if len(values_to_smooth) > 0 else 1

            smoothed_values[iteration] = accumulator
            values_to_smooth.clear()

        return smoothed_values, list(sorted(smoothed_values.keys()))

    def plot(self, start=0, end=None, smooth_values=False):
        self.parse_log_file(start=start, end=end)

        metrics_to_plot = sorted(next(iter(self.train_iterations.values())).keys(), key=lambda x: x.rsplit('_'))
        fig, axes = plt.subplots(len(metrics_to_plot), sharex=True)

        if smooth_values:
            train_iterations, x_train = self.smooth_values(self.train_iterations)
            test_iterations, x_test = self.smooth_values(self.test_iterations)
        else:
            x_train = list(sorted(self.train_iterations.keys()))
            x_test = list(sorted(self.test_iterations.keys()))
            train_iterations = self.train_iterations
            test_iterations = self.test_iterations

        for metric, axe in zip(metrics_to_plot, axes):
            axe.plot(x_train, [train_iterations[iteration][metric] for iteration in x_train], 'r.-', label='train')
            axe.plot(x_test, [test_iterations[iteration][metric] for iteration in x_test], 'g.-', label='test')

            axe.set_title(metric)

            box = axe.get_position()
            axe.set_position([box.x0, box.y0, box.width * 0.9, box.height])

            axe.legend(bbox_to_anchor=(1, 0.5), loc='center left', fancybox=True, shadow=True)

        return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool to create plots of training')
    parser.add_argument("log_file", help="path to log file")
    parser.add_argument("-d", "--destination", dest='destination', help='where to save the resulting plot')
    parser.add_argument("-f", "--from", dest='start', default=0, type=int, help="start index from which you want to plot")
    parser.add_argument("-t", "--to", dest='end', type=int, help="index until which you want to plot (default: end)")
    parser.add_argument("-s", "--smooth", dest='smooth', action='store_true', default=False, help="smooth logged metrics over course of 1000 iterations")

    args = parser.parse_args()

    plotter = LogPlotter(args.log_file)
    fig = plotter.plot(start=args.start, end=args.end, smooth_values=args.smooth)
    if args.destination is None:
        plt.show()
    else:
        fig.savefig(args.destination)