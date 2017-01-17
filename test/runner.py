from rsk.rsk import RSK
from rsk.panel import PanelSeries
import scipy as sp
from scipy import matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":
    panel_series = PanelSeries.from_csv("private/ebola.csv", 0, 1)
    print("Analysis of mVAMPrices Ebola data: ")
    print("VARS: %s" % str(panel_series.variable_names))
    print("GROUPS: %s" % str(panel_series.groups))
    print("TIMES: %s" % str(panel_series.times))
    print()
    print()
    n_vars = 4
    n_groups = 3

    # Specify a slow moving transition matrix for alpha
    n_alpha = 64        # for this implementation, we need n_alpha>n_groups*n_vars
    delta = 0.01       # off diagonal transition probability
    transition_matrix = (1-(n_alpha-1)*delta)*sp.eye(n_alpha) + (delta*sp.ones((n_alpha,n_alpha)) - delta*sp.eye(n_alpha))

    # random translation matrix
    translation_matrix = matrix(sp.random.uniform(size=(n_groups*n_vars, n_alpha)))

    # solve for a0 so that the estimated mean at time 0 is mu[0]
    means = panel_series.means()
    a0 = sp.linalg.pinv(translation_matrix).dot(means[0].reshape(-1,1))  # solve via generalized inverse since this system is underdetermined
    a0 = sp.linalg.solve(transition_matrix, matrix(a0))

    # specify an alpha covariance structure
    Q =  matrix(sp.eye(n_alpha))
    Q0 = matrix(sp.eye(n_alpha))

    # fit RSK
    rsk_filter = RSK(transition_matrix, translation_matrix)
    fitted_means = rsk_filter.fit(panel_series, a0, Q0, Q, smooth=True, sigma = sp.eye(n_vars))


    guinea, liberia, sierra_leone = [],[],[]
    rsk_guinea, rsk_liberia, rsk_sierra_leone = [],[],[]
    for i,true_mean, rsk_mean in zip(panel_series.times, means,fitted_means):
        print("At time %d: "  % i)
        print("Naive sample mean: \n%s" % true_mean)
        print("RSK mean: \n%s" % rsk_mean)

        guinea.append(true_mean[0])
        rsk_guinea.append(rsk_mean[0])

        liberia.append(true_mean[1])
        rsk_liberia.append(rsk_mean[1])

        sierra_leone.append(true_mean[2])
        rsk_sierra_leone.append(rsk_mean[2])

    guinea = sp.vstack(guinea)
    rsk_guinea = sp.vstack(rsk_guinea)

    liberia = sp.vstack(liberia)
    rsk_liberia = sp.vstack(rsk_liberia)

    sierra_leone = sp.vstack(sierra_leone)
    rsk_sierra_leone = sp.vstack(rsk_sierra_leone)

    plt.close('all')
    f, axarr = plt.subplots(3, 4, sharex=True)
    x = panel_series.times

    # guinea
    axarr[0, 0].plot(x, guinea[:,0])
    axarr[0, 0].plot(x, rsk_guinea[:,0], c="red")
    axarr[0,0].set_ylabel("Guinea\nprice\n")
    axarr[0,0].set_title(panel_series.variable_names[2])

    axarr[0, 1].plot(x, guinea[:,1])
    axarr[0, 1].plot(x, rsk_guinea[:,1], c="red")
    axarr[0, 1].set_title(panel_series.variable_names[3])

    axarr[0, 2].plot(x, guinea[:,2])
    axarr[0, 2].plot(x, rsk_guinea[:,2], c="red")
    axarr[0, 2].set_title(panel_series.variable_names[4])

    axarr[0, 3].plot(x, guinea[:,3])
    axarr[0, 3].plot(x, rsk_guinea[:,3], c="red")
    axarr[0, 3].set_title(panel_series.variable_names[5])

    # liberia
    axarr[1, 0].plot(x, liberia[:,0])
    axarr[1, 0].plot(x, rsk_liberia[:,0], c="red")
    axarr[1,0].set_ylabel("Liberia\nPrice\n")

    axarr[1, 1].plot(x, liberia[:,1])
    axarr[1, 1].plot(x, rsk_liberia[:,1], c="red")


    axarr[1, 2].plot(x, liberia[:,2])
    axarr[1, 2].plot(x, rsk_liberia[:,2], c="red")

    axarr[1, 3].plot(x, liberia[:,3])
    axarr[1, 3].plot(x, rsk_liberia[:,3], c="red")

    # sierra leone
    axarr[2, 0].plot(x, sierra_leone[:,0])
    axarr[2, 0].plot(x, rsk_sierra_leone[:,0], c="red")
    axarr[2,0].set_ylabel("Sierra Leone\nPrice")

    axarr[2, 1].plot(x, sierra_leone[:,1])
    axarr[2, 1].plot(x, rsk_sierra_leone[:,1], c="red")

    axarr[2, 2].plot(x, sierra_leone[:,2])
    axarr[2, 2].plot(x, rsk_sierra_leone[:,2], c="red")

    axarr[2, 3].plot(x, sierra_leone[:,3])
    axarr[2, 3].plot(x, rsk_sierra_leone[:,3], c="red")

    plt.show()