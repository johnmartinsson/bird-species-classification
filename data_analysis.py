import xml.etree.ElementTree as ET
import glob
import os
import numpy
import tqdm
import optparse
import matplotlib.pyplot as plt
from scipy.stats import norm

# from bird import analysis
from bird import loader

parser = optparse.OptionParser()
parser.add_option("--xml_dir", dest="xml_dir")
(options, args) = parser.parse_args()

def represents_float(text):
    try:
        float(text)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

def represents_int(text):
    try:
        int(text)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

# def elevation_bins(class_id, completes):
    # for c in completes:
        # if c['ClassId'] = class_id

def groupby(xs, func):
    groups = {}
    for x in xs:
        if func(x) not in groups:
            groups[func(x)] = [x]
        else:
            groups[func(x)].append(x)
    return groups.items()

def plot_elevation_histograms(completes):
    groups = groupby(completes, lambda x: x['ClassId'])
    fig = plt.figure(1)
    for key, group in groups:
        elevations = [v['Elevation'] for v in group]
        plt.hist(elevations, bins=40, range=(0, 4500))
        plt.title("Class: {} (mean = {}, std = {})".format(key,
                                                             int(numpy.mean(elevations)),
                                                             int(numpy.std(elevations))))
        plt.xlabel("Elevation [m]")
        plt.ylabel("Observations")
        fig.savefig(os.path.join("histograms", key + ".png"))
        fig.clf()

def plot_date_histograms(xml_roots):
    data = [{
        'ClassId':r.find("ClassId").text
        , 'Month':int(r.find("Date").text.split("-")[1])
    } for r in xml_roots]
    groups = groupby(data, lambda x: x['ClassId'])
    fig = plt.figure(1)
    for key, group in groups:
        months = [v['Month'] for v in group]
        plt.hist(months, bins=12, range=(0, 12))
        plt.title("Class: {} ".format(key))
        plt.xlabel("Month")
        plt.ylabel("Observations")
        fig.savefig(os.path.join("date_histograms", key + ".png"))
        fig.clf()

def plot_time_histograms(xml_roots):
    def parse(time):
        v = None
        try:
            v = int(time.split(":")[0])
        except ValueError:
            v = -1
        except AttributeError:
            v = -1
        return v
    data = [{
        'ClassId':r.find("ClassId").text
        , 'Time':parse(r.find("Time").text)
    } for r in xml_roots]
    groups = groupby(data, lambda x: x['ClassId'])
    fig = plt.figure(1)
    for key, group in groups:
        months = [v['Time'] for v in group]
        plt.hist(months, bins=24, range=(0, 24))
        plt.title("Class: {} ".format(key))
        plt.xlabel("Time")
        plt.ylabel("Observations")
        fig.savefig(os.path.join("time_histograms", key + ".png"))
        fig.clf()

def segments_to_training_files(training_segments):
    training_files = ["_".join(s.split("_")[:5]) + ".wav" for s in training_segments]
    training_files = list(set(training_files))
    training_files = [os.path.basename(f) for f in training_files]
    return training_files

def build_elevation_distributions(xml_roots, train_dir):
    training_segments = glob.glob(os.path.join(train_dir, "*", "*.wav"))
    training_files = segments_to_training_files(training_segments)

    elevation_observations = {}

    index_to_species = loader.build_class_index(train_dir)
    species_to_index = {v : k for (k, v) in index_to_species.items()}
    nb_classes = len(index_to_species.items())

    for r in xml_roots:
        file_name = r.find("FileName").text
        elevation = r.find("Elevation").text
        if file_name in training_files and represents_int(elevation):
            class_id = r.find("ClassId").text
            # if species_to_index[class_id] == 806:
                # print(file_name)
            if class_id in elevation_observations:
                elevation_observations[class_id].append(int(elevation))
            else:
                elevation_observations[class_id] = [int(elevation)]

    def gpd(mu, sigma, max_elevation, nb_observations):
        weight = 1
        if nb_observations < 10:
            weight = 1
        else:
            weight = 1/nb_observations

        return lambda x: ((1-weight) * norm.pdf(x, mu, sigma) + weight * (1/max_elevation))/2

    max_elevation = 5000
    elevation_to_probability = {}
    for class_id, elevations in elevation_observations.items():
        # print(class_id, elevations)
        mu = numpy.mean(elevations)
        sigma = numpy.std(elevations)
        if sigma == 0.0:
            elevation_to_probability[class_id] = lambda x: 1/max_elevation
        else:
            elevation_to_probability[class_id] = gpd(mu, sigma, max_elevation,
                                                     len(elevations))

        # if species_to_index[class_id] == 806:
            # print("index:", species_to_index[class_id], "mean:", mu, "std:", sigma)
            # print(elevations)

    # print(species_to_index)
    elevation_to_probability = {species_to_index[k] : v for (k, v) in
                                elevation_to_probability.items()}

    return elevation_to_probability


def get_completes(xml_roots):
    completes = []
    for r in xml_roots:
        lat = r.find("Latitude").text
        lon = r.find("Longitude").text
        ele = r.find("Elevation").text
        class_id = r.find("ClassId").text
        if represents_float(lat) and represents_float(lon) and represents_int(ele):
            obj = {
                "ClassId": class_id
                , "Latitude": float(lat)
                , "Longitude": float(lon)
                , "Elevation": int(ele)
            }
            completes.append(obj)
    return completes

def load_xml_roots(xml_dir):
    xml_paths = glob.glob(os.path.join(xml_dir, "*.xml"))
    print("loading xml data ...")
    progress = tqdm.tqdm(range(len(xml_paths)))
    xml_roots = [ET.parse(f) for (p, f) in zip(progress,
                                                                  xml_paths)]
    return xml_roots

if __name__ == "__main__":
    main()
