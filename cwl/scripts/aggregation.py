def init_params(aggregated_net, nets):
    '''Init layer parameters.'''
    aggregate_sd = aggregated_net.state_dict()
    sds = [n.state_dict() for n in nets]
    for key in aggregate_sd:
        aggregate_sd[key] = sum([sd[key] for sd in sds]) / len(nets)
    aggregated_net.load_state_dict(aggregate_sd)
