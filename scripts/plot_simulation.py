#plot a short simulation
@timed.namedtimer("plot")
def plot(filename, t, address_input, pre_decoded, cleanup_spikes,
          output_decoded, output_sim, input_sim, **kwargs):

    num_plots = 6
    offset = num_plots * 100 + 10 + 1

    ax, offset = nengo_plot_helper(offset, t, address_input)
    ax, offset = nengo_plot_helper(offset, t, pre_decoded)
    ax, offset = nengo_plot_helper(offset, t, cleanup_spikes, spikes=True)
    ax, offset = nengo_plot_helper(offset, t, output_decoded)
    ax, offset = nengo_plot_helper(offset, t, output_sim)
    ax, offset = nengo_plot_helper(offset, t, input_sim)

    plt.savefig(filename)
