

def load_events(filename, time_factor=1, start_factor=1):
    res = []
    with open(filename, "r") as in_file:
        first = next(in_file)
        values_first = first.split(" ")
        for line in in_file:
            values = line.split(" ")
            res.append((float(values[0]) * time_factor, int(values[1])))
    return (float(values_first[0]), float(values_first[1]) * start_factor), res