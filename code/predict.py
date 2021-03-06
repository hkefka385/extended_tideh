from fun


def brent_search_minimize(change_point, fitting_arg, events, obs_time):
    events_ = [e for e in events if e[0] < obs_time]

    U1 = []
    for i, event in enumerate(events_):
        t = event[0]
        fu_ = 0
        for j in range(i):
            if events_[j][0] > change_point:
                continue
            inp = (t - events_[j][0])
            fu_ += kernel_zhao(inp) * events_[j][1]
        U1.append(fu_)

    # calculation U2
    events_af =  [e for e in events_ if e[0] >= change_point]
    U2 = []
    for i, event in enumerate(events_af):
        t = event[0]
        if t == 0.0:
            continue
        fu_ = 0
        for j in range(i):
            inp = (t - events_af[j][0])
            fu_ += kernel_zhao(inp) * events_af[j][1]
        U2.append(fu_)

    arg = (events, obs_time, change_point, U1, U2)
    mle = minimize(MLE_method_fake_bret, fitting_arg, args = arg ,method='Nelder-Mead')

    return mle


def MLE_method_fake_tweet(fitting, events, obs_time, change_point):
    events_ = [e for e in events if e[0] < obs_time]
    events_be = [e for e in events_ if e[0] < change_point]
    events_af = [e for e in events_ if e[0] >= change_point]

    n = len(events_)
    be_n = len(events_be)
    be_a = len(events_af)

    p0 = fitting[0]
    r0 = fitting[1]
    phi0 = fitting[2]
    taum = fitting[3]

    p0_a = fitting[4]
    r0_a = fitting[1]
    phi0_a = fitting[2]
    taum_a = fitting[5]

    kkk = []
    fj = 0
    for i in range(math.ceil(change_point)):
        if change_point - 1 < i:
            t_start = i
            t_end = change_point
            event_obs = [e for e in events_ if e[0] < t_end]
            F = estimate_K(event_obs, t_start, t_end, integral_zhao)
            s = (t_start + t_end) / 2
            est = infectious_rate_tweets(t=s, p0=p0, r0=r0, phi0=phi0, taum=taum)
            fj += F * est
        else:
            t_start = i
            t_end = i + 1
            event_obs = [e for e in events_ if e[0] < t_end]
            F = estimate_K(event_obs, t_start, t_end, integral_zhao)
            s = (t_start + t_end) / 2
            est = infectious_rate_tweets(t=s, p0=p0, r0=r0, phi0=phi0, taum=taum)
            fj += F * est
            kkk.append(fj)

    for i in range(math.floor(change_point), obs_time):
        if i == math.floor(change_point):
            t_start = change_point
            t_end = i + 1
            event_obs1 = [e for e in events_ if e[0] < t_end]
            event_obs2 = [e for e in events_ if e[0] < t_end and change_point <= e[0]]
            F1 = estimate_K(event_obs1, t_start, t_end, integral_zhao)
            F2 = estimate_K(event_obs2, t_start, t_end, integral_zhao)
            s = (t_start + t_end) / 2
            est1 = infectious_rate_tweets(t=s, p0=p0, r0=r0, phi0=phi0, taum=taum)
            est2 = infectious_rate_tweets(t=s, p0=p0_a, r0=r0_a, phi0=phi0_a, taum=taum_a)
            fj += F1 * est1 + F2 * est2
            kkk.append(fj)
        else:
            t_start = i
            t_end = i + 1
            event_obs1 = [e for e in events_ if e[0] < t_end]
            event_obs2 = [e for e in events_ if e[0] < t_end and change_point <= e[0]]
            F1 = estimate_K(event_obs1, t_start, t_end, integral_zhao)
            F2 = estimate_K(event_obs2, t_start, t_end, integral_zhao)
            s = (t_start + t_end) / 2
            est1 = infectious_rate_tweets(t=s, p0=p0, r0=r0, phi0=phi0, taum=taum)
            est2 = infectious_rate_tweets(t=s, p0=p0_a, r0=r0_a, phi0=phi0_a, taum=taum_a)
            fj += F1 * est1 + F2 * est2
            kkk.append(fj)

    return fj, kkk