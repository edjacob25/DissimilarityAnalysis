import argparse
import multiprocessing
import subprocess
import time
import traceback
from dataclasses import astuple
from itertools import product
from shutil import copyfile
from typing import Tuple, Optional

import git
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from common import *
from data_types import *


def remove_attribute(filepath: Path, attribute: str):
    attributes = []
    data = []
    permitted_indexes = []
    with filepath.open("r") as file:
        relation_name = file.readline()
        data_processing = False
        i = 0
        for line in file:
            if not data_processing:
                line_upper = line.upper()
                if line_upper.startswith("@ATTRIBUTE"):
                    if attribute.upper() != line_upper.split(" ")[1]:
                        attributes.append(line)
                        permitted_indexes.append(i)
                    i += 1
                    continue
                if line_upper.startswith("@DATA"):
                    data_processing = True
            else:
                line_data = [x.lstrip() for x in line.split(",")]
                data.append(line_data)

    with filepath.open("w") as new_file:
        new_file.write(relation_name)
        for item in attributes:
            new_file.write(item)
        new_file.write("\n")
        new_file.write("@DATA\n")
        for datapoint in data:
            points = [x for i, x in enumerate(datapoint) if i in permitted_indexes]
            result = ",".join(points)
            new_file.write(result)


# TODO: Add option to read classpath from the config file
def cluster_dataset(
    exp_parameters: ExperimentParameters,
    set_parameters: ExperimentSetParameters,
    parameters: GeneralParameters,
    start_mode: str = "1",
) -> Experiment:
    _, measure, strategy, weight, learning_based, auc_type = astuple(exp_parameters)
    filepath = exp_parameters.filepath
    clustered_name = filepath.name.replace(".arff", f"_{measure}_{strategy}_{weight}_{auc_type}_clustered.arff")
    clustered_file_path = filepath.with_name(clustered_name)

    command = ["java", "-Xmx8192m"]

    if parameters.classpath is not None:
        java_classpath = parameters.classpath
    else:
        separator = get_platform_separator()
        java_classpath = (
            f"{get_config('ROUTES', 'weka_jar_path')}{separator}"
            f"{get_config('ROUTES', 'autoweka_jar_path')}{separator}"
            f"{get_config('ROUTES', 'dissimilarity_jar_path')}"
        )
    command.append("-cp")
    command.append(java_classpath)

    command.append("weka.filters.unsupervised.attribute.AddCluster")
    command.append("-W")

    num_classes = get_number_of_clusters(filepath)
    if parameters.verbose:
        print(f"Number of clusters for {filepath} is {num_classes}")
    num_procs = multiprocessing.cpu_count()
    if learning_based:
        distance_function = f"weka.core.{measure} -R first-last -S {strategy} -W {weight}"
        if parameters.normalize_dissimilarity:
            distance_function = f"{distance_function} -n"
    else:
        distance_function = f"weka.core.{measure}"

    if exp_parameters.auc_type is not None:
        distance_function = f"{distance_function} -a {auc_type}"

    if not set_parameters.initial:
        distance_function = (
            f'"{distance_function} -w {set_parameters.weight} -o {set_parameters.strategy} '
            f'-t {set_parameters.multiplier}"'
        )
    else:
        distance_function = f'"{distance_function}"'

    clusterer = (
        f"weka.clusterers.CategoricalKMeans -init {start_mode} -max-candidates 100 -periodic-pruning 10000 "
        f"-min-density 2.0 -t1 -1.25 -t2 -1.0 -N {num_classes} -A {distance_function} -I 500 "
        f"-num-slots {math.floor(num_procs / 3)} -S 10"
    )
    command.append(clusterer)
    command.append("-i")
    command.append(str(filepath))
    command.append("-o")
    command.append(str(clustered_file_path))
    command.append("-I")
    command.append("Last")

    start_dt = datetime.now()
    start = time.time()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end = time.time()

    identifier = f"measure {measure}"
    if exp_parameters.learning_based:
        identifier = f"{identifier}(strategy: {strategy}, weight: {weight})"
    if start_mode == "0":
        identifier = f"{identifier} classic mode"

    write_results(filepath, result)
    errors = result.stderr.decode("utf-8").splitlines()
    if len(errors) > 40:
        print(f"{fg.orange}Errors too long, showing the last 40 lines:{fg.rs}")
        print("\n".join(errors[-40:]))
    else:
        print(f"{fg.orange}Showing errors:{fg.rs}")
        print("\n".join(errors))
    if parameters.verbose:
        print(result.stdout.decode("utf-8"))
        print(f"Clustering dataset {filepath} with {identifier} took {end - start}")

    if clustered_file_path.stat().st_size >= filepath.stat().st_size:
        remove_attribute(clustered_file_path, "Class")
        number_of_clusters = get_number_of_clusters(clustered_file_path)
        print(f"Finished clustering dataset {filepath} with {identifier}")
    else:
        if clustered_file_path.exists():
            clustered_file_path.unlink()

        if start_mode == "1":
            print(
                f"{fg.orange}There was an error running weka in file {filepath.name} with k-means++ mode, "
                f"trying with classic mode{fg.rs}"
            )
            return cluster_dataset(exp_parameters, set_parameters, parameters, start_mode="0")
        else:
            raise Exception(
                f"{fg.red}There was a error running weka with the file {filepath.name} and the"
                + f" following command {' '.join(result.args)}{fg.rs}"
            )

    if start_mode == "1":
        return Experiment(
            method=distance_function.replace('"', ""),
            command_sent=" ".join(command),
            time_taken=end - start,
            k_means_plusplus=True,
            file_name=filepath.name,
            number_of_classes=num_classes,
            number_of_clusters=number_of_clusters,
            start_time=start_dt,
            comments="",
        )
    else:
        return Experiment(
            method=distance_function.replace('"', ""),
            command_sent=" ".join(command),
            time_taken=end - start,
            k_means_plusplus=False,
            file_name=filepath.name,
            number_of_classes=num_classes,
            number_of_clusters=number_of_clusters,
            start_time=start_dt,
            comments="",
        )


def copy_files(exp_params: ExperimentParameters) -> Tuple[Path, Path]:
    _, measure, strategy, weight, learning_based, auc = astuple(exp_params)
    filepath = exp_params.filepath
    path: Path = filepath.parent
    file, filename = filepath.name, filepath.stem

    if learning_based:
        new_folder_path = path / f"{filename}_{measure}_{strategy}_{weight}_{auc}"
    else:
        new_folder_path = path / f"{filename}_{measure}"

    new_folder_path.mkdir()
    new_filepath = new_folder_path / file
    copyfile(str(filepath), new_filepath)

    old_clustered_filepath = path / f"{filename}_{measure}_{strategy}_{weight}_{auc}_clustered.arff"
    new_clustered_filepath = new_folder_path / f"{filename}.clus"
    copyfile(str(old_clustered_filepath), new_clustered_filepath)
    old_clustered_filepath.unlink()
    return new_filepath, new_clustered_filepath


def do_single_experiment(
    item: Path,
    strategy: str,
    weight: str,
    set_parameters: ExperimentSetParameters,
    params: GeneralParameters,
    auc_type: str = None,
) -> Experiment:
    if weight is None:
        exp_params = ExperimentParameters(filepath=item, measure=strategy)
    else:
        exp_params = ExperimentParameters(
            filepath=item,
            measure="LearningBasedDissimilarity",
            strategy=strategy,
            weight_strategy=weight,
            learning_based=True,
            auc_type=auc_type,
        )

    exp = cluster_dataset(exp_params, set_parameters, params)

    new_filepath, new_clustered_filepath = copy_files(exp_params)

    f_measure = get_measure(
        new_filepath,
        new_clustered_filepath,
        exe_path=params.measure_calculator_path,
        verbose=params.verbose,
    )
    exp.f_score = f_measure
    return exp


def do_experiment_set(set_params: ExperimentSetParameters, params: GeneralParameters):
    if set_params.alternate:

        measures = [
            "Eskin",
            "Gambaryan",
            "Goodall",
            "Lin",
            "OccurenceFrequency",
            "InverseOccurenceFrequency",
            "EuclideanDistance",
            "ManhattanDistance",
            "LinModified",
            "LinModified_Kappa",
            "LinModified_MinusKappa",
            "LinModified_KappaMax",
        ]
        if not set_params.initial:
            measures = [
                "EskinModified",
                "GambaryanModified",
                "GoodallModified",
                "OFModified",
                "IOFModified",
                "LinModified",
            ]

        measures = list(zip(measures, [None for _ in range(len(measures))]))
    else:
        strategies = ["A", "B", "C", "D", "E", "N"]
        weights = ["N", "K", "A"]
        measures = list(product(strategies, weights))

    engine = create_engine("sqlite:///Results/results.db")
    Base.metadata.create_all(engine)
    session_class = sessionmaker(bind=engine)
    session = session_class()
    code_directory = Path.cwd().parent
    repo = git.Repo(code_directory)

    description = f"{set_params.description}"
    if not set_params.initial:
        description = (
            f"{description} using {set_params.weight} as decision weight, doing {set_params.strategy} when "
            f"weight is low and multiplying by {set_params.multiplier}"
        )

    exp_set = ExperimentSet(
        time=datetime.now(),
        base_directory=str(params.directory),
        commit=repo.head.object.hexsha,
        description=description,
    )
    session.add(exp_set)
    session.commit()

    start = time.time()
    i = 0
    pool = multiprocessing.Pool(math.floor(multiprocessing.cpu_count() / 2))
    for item in params.directory.iterdir():
        if item.suffix == ".arff" and "clustered" not in item.stem:
            try:
                sets = pool.starmap(
                    do_single_experiment,
                    [(item, strategy, weight, set_params, params) for strategy, weight in measures],
                )
                exp_set.experiments.extend(sets)
                session.commit()
                i += 1
            except KeyboardInterrupt:
                session.rollback()
                print(f"{fg.orange}The analysis of the file {item} was requested to be finished by using Ctrl-C{fg.rs}")
                continue
            except Exception as exc:
                session.rollback()
                print(exc)
                print(f"{fg.red}Skipping file {item}{fg.rs}")
                continue
            finally:
                print("\n\n")

    end = time.time()

    exp_set.time_taken = end - start
    exp_set.number_of_datasets = i
    session.commit()

    time_str = format_time_lapse(start, end)
    send_notification(f"It took {time_str} and processed {i} datasets", "Analysis finished")


def full_experiments(params: GeneralParameters):
    clean_experiments(params.directory)
    set_params = ExperimentSetParameters(initial=True, description="Initial Learning Based")
    do_experiment_set(set_params, params)
    set_params.alternate = True
    set_params.description = "Initial Other Measures"
    do_experiment_set(set_params, params)
    save_results(params.directory, f"Results_initial.zip")
    clean_experiments(params.directory)

    weights = ["K", "A"]
    strategies = ["B", "D", "M", "L"]
    multipliers = ["N", "I", "O"]
    for weight in weights:
        for strategy in strategies:
            for multiplier in multipliers:
                set_params = ExperimentSetParameters(
                    strategy=strategy,
                    multiplier=multiplier,
                    weight=weight,
                    initial=False,
                    description="Learning Based",
                )
                do_experiment_set(set_params, params)
                set_params.alternate = True
                set_params.description = "Modified Measures"
                do_experiment_set(set_params, params)
                results_archive_name = f"Results_weight{weight}_strategy{strategy}_multiply_{multiplier}.zip"
                if params.normalize_dissimilarity:
                    results_archive_name = results_archive_name.replace(".zip", "_n.zip")
                save_results(params.directory, results_archive_name)
                clean_experiments(params.directory)


def do_experiment_guarded(
    item: Path,
    strategy: str,
    weight: str,
    set_parameters: ExperimentSetParameters,
    params: GeneralParameters,
    auc_type: str = None,
) -> Optional[Experiment]:
    try:
        return do_single_experiment(item, strategy, weight, set_parameters, params, auc_type)
    except KeyboardInterrupt:
        print(f"{fg.orange}The analysis of the file {item} was requested to be finished by using Ctrl-C{fg.rs}")
        return None
    except Exception:
        traceback.print_exc()
        print(f"{fg.orange}Skipping file {item}{fg.rs}")
        return None
    finally:
        print("\n\n")


def do_auc_exps(params: GeneralParameters):
    auc_types = ["N", "S", "W"]

    engine = create_engine("sqlite:///Results/results_auc.db")
    Base.metadata.create_all(engine)
    session_class = sessionmaker(bind=engine)
    session = session_class()
    code_directory = Path.cwd().parent
    repo = git.Repo(code_directory)

    description = f"Testing dissimilarity AUCs "

    exp_set = ExperimentSet(
        time=datetime.now(),
        base_directory=str(params.directory),
        commit=repo.head.object.hexsha,
        description=description,
    )
    session.add(exp_set)
    session.commit()
    set_params = ExperimentSetParameters(initial=True, description="Learning Based")
    start = time.time()
    i = 0
    pool = multiprocessing.Pool(math.floor(multiprocessing.cpu_count() / 2))
    for item in params.directory.iterdir():
        if item.suffix == ".arff" and "clustered" not in item.stem:
            try:
                sets = pool.starmap(
                    do_single_experiment,
                    [(item, "E", "K", set_params, params, auc) for auc in auc_types],
                )
                exp_set.experiments.extend(sets)
                session.commit()
                i += 1
            except KeyboardInterrupt:
                session.rollback()
                print(f"{fg.orange}The analysis of the file {item} was requested to be finished by using Ctrl-C{fg.rs}")
                continue
            except Exception as exc:
                session.rollback()
                print(exc)
                print(f"{fg.red}Skipping file {item}{fg.rs}")
                continue
            finally:
                print("\n\n")

    end = time.time()

    exp_set.time_taken = end - start
    exp_set.number_of_datasets = i
    session.commit()

    time_str = format_time_lapse(start, end)
    send_notification(f"It took {time_str} and processed {i} datasets", "Analysis finished")


def do_selected_exps(params: GeneralParameters):
    clean_experiments(params.directory)
    engine = create_engine("sqlite:///Results/results_training_input.db")
    Base.metadata.create_all(engine)
    session_class = sessionmaker(bind=engine)
    session = session_class()

    measures = [
        "Eskin",
        "Gambaryan",
        "Goodall",
        "Lin",
        "OccurenceFrequency",
        "InverseOccurenceFrequency",
        "EuclideanDistance",
        "ManhattanDistance",
        "LinModified_Kappa",
        "LinModified_KappaMax",
    ]
    measures = list(zip(measures, [None for _ in range(len(measures))]))
    measures.append(("E", "N"))
    description = f"Mixed datasets with selected algorithms"

    exp_set = ExperimentSet(
        time=datetime.now(),
        base_directory=str(params.directory),
        commit="",
        description=description,
    )
    session.add(exp_set)
    session.commit()
    set_params = ExperimentSetParameters(initial=True, description=description)
    start = time.time()
    items = [x for x in params.directory.iterdir() if x.suffix == ".arff" and "clustered" not in x.stem]

    pool = multiprocessing.Pool(6, init_worker)
    results = []

    try:
        exps = [(item, measure, weight, set_params, params) for item in items for measure, weight in measures]
        sets = pool.starmap(do_experiment_guarded, exps)
        results = [x for x in sets if x is not None]
        exp_set.experiments.extend(results)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        kill_java_procs()
        print(f"{fg.orange}Ctrl-C used, stopped job{fg.rs}")
    session.commit()
    end = time.time()

    exp_set.time_taken = end - start
    exp_set.number_of_datasets = len(results)
    session.commit()
    time_str = format_time_lapse(start, end)
    send_notification(f"It took {time_str} and processed {len(results)} datasets", "Analysis finished")


def main():
    parser = argparse.ArgumentParser(description="Does the analysis of a directory containing categorical datasets")
    parser.add_argument("directory", help="Directory in which the cleaned datasets are")
    parser.add_argument(
        "-cp",
        help="Classpath for the weka invocation, needs to contain the weka.jar file and probably the jar of the "
        "measure ",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Show the output of the weka commands",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--measure-calc",
        help="Path to the f-measure calculator",
        dest="measure_calculator_path",
    )
    parser.add_argument(
        "--alternate-analysis",
        help="Does the alternate analysis with the already known simmilarity measures",
        action="store_true",
    )
    parser.add_argument("-s", "--save", help="Path to the f-measure calculator", action="store_true")

    parser.add_argument(
        "--full",
        help="Whether to do all combinations, overrides --alternate-analysis",
        action="store_true",
    )
    parser.add_argument(
        "--selected",
        help="Whether to do only a set of the experiments, overrides --alternate-analysis",
        action="store_true",
    )
    # TODO: Actually save the output of the commands

    args = parser.parse_args()

    # TODO: Read a single file
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"{fg.red}The selected path is not a directory{fg.rs}")
        exit(1)

    directory = directory.resolve()
    params = GeneralParameters(
        directory=directory,
        verbose=args.verbose,
        classpath=args.cp,
        measure_calculator_path=args.measure_calculator_path,
    )
    if args.full:
        print(f"{fg.red}DOING ALL EXPERIMENTS{fg.rs}")
        print(f"{fg.green}Go for a coffee{fg.rs}")
        full_experiments(params)

    elif args.selected:
        print(f"{fg.green}Doing selected experiments{fg.rs}")
        do_selected_exps(params)
    else:
        results_archive_name = f"Results_single_alt_{args.alternate_analysis}.zip"
        save_results(params.directory, results_archive_name)
        clean_experiments(params.directory)
        set_params = ExperimentSetParameters(alternate=args.alternate_analysis)
        do_experiment_set(set_params, params)


if __name__ == "__main__":
    main()
