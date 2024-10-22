import numpy as np
import matplotlib.pyplot as plt

mog_marker_list = ['o', 's', '^', 'x']  # Circle, square, triangle up, x
colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # bright orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # dark orange
    "#e377c2",  # magenta
    "#bcbd22",  # yellowish green
    "#dbdb8d",  # light beige
    "#17becf",  # blue-green
    "#9edae5",  # light blue
    "#ffffcc",  # light yellow
    "#c7c7c7",  # dark gray
    "#f7cac9",  # light pink
    "#fc8d59",  # salmon pink
    "#7f7f7f",  # gray (included for completeness, consider using for background)
]

class OnlineToyDataset():
    def __init__(self, data_name):
        super().__init__()
        
        self.data_name = data_name
        self.rng = np.random.RandomState(42)

    def gen_batch(self, batch_size):
        return inf_train_gen(self.data_name, self.rng, batch_size)

    def get_category_sizes(self):
        if self.data_name == "rings":
            return [4, len(colors)]
        else:
            raise NotImplementedError
    
    def get_numerical_sizes(self):
        return 2

def inf_train_gen(data, rng=None, batch_size=4096):

    if data == "rings":
        from scipy.stats import truncnorm
        def assign_color_labels(x):
            num_classes = 8
            class_width = 1.0 / num_classes
            label = x // class_width
            return label.astype(int)

        toy_radius = 1.5
        toy_sd = 0.015
        truncnorm_rv = truncnorm(
                a=(0 - toy_radius) / toy_sd,
                b=np.inf,
                loc=toy_radius,
                scale=toy_sd,
            )
        
        rnd_theta = np.random.random(batch_size)
        groups_label = np.random.randint(0, 4, batch_size)
        color_label = assign_color_labels(rnd_theta)
        color_label = (color_label + groups_label * 4) % len(colors)

        sample_radii = truncnorm_rv.rvs(batch_size)
        # sample_radii = sample_radii * (groups_label + 1)
        sample_thetas = 2 * np.pi * rnd_theta
        sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(-1, 1)
        sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(-1, 1)

        sample_group = np.concatenate((sample_x, sample_y), axis=1)

        data = np.zeros((batch_size, 4))
        data[:, 0] = sample_group[:, 0]
        data[:, 1] = sample_group[:, 1]
        data[:, 2] = groups_label
        data[:, 3] = color_label
        return data

    else:
        raise NotImplementedError

def plot_rings_example(data, save_path='data_samples.png'):
    left_bound = -7
    right_bound = 7
    
    group_label = data[:, 2]
    color_label = data[:, 3]
    plt.scatter(data[:, 0]*(group_label+1), data[:, 1]*(group_label+1), s=0.2, c=[colors[int(i)] for i in color_label], marker='o')

    plt.axis('square')
    plt.axis('off')
    # plt.title('data samples')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    data = inf_train_gen("rings", rng, 10000)
    plot_rings_example(data, 'rings_data.png')

