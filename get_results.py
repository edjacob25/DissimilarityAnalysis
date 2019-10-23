import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from common import *
from data_types import *


def main():
    parser = argparse.ArgumentParser(description='Obtains ans saves the results of several directories containing '
                                                 'classified categorical datasets')
    parser.add_argument('directory', help="Directory in which the cleaned datasets are")
    parser.add_argument('--rand', help="Use rand index instead of F-Score", action='store_true')
    parser.add_argument('--adjusted-rand', help="Use adjusted rand index instead of F-Score", action='store_true')
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"{fg.red}The selected path is not a directory{fg.rs}")
        exit(1)

    directory = directory.resolve()
    params = GeneralParameters(directory=directory, verbose=False)

    engine = create_engine('sqlite:///Results/results_rand.db')
    Base.metadata.create_all(engine)
    session_class = sessionmaker(bind=engine)
    session = session_class()
    description = f"Testing datasets with selected algorithms"

    exp_set = ExperimentSet(time=datetime.now(), base_directory=str(params.directory), commit="",
                            description=description)
    session.add(exp_set)
    session.commit()
    dirs = [x for x in directory.iterdir() if x.is_dir()]
    for directory in dirs:
        base, clustered = directory.iterdir()
        measure = get_measure(base, clustered, rand=args.rand, adjusted_rand=args.adjusted_rand)
        distance = directory.name.replace(f"{base.stem}_", " ").replace("_", " ")
        num_classes = get_number_of_clusters(base)
        number_of_clusters = get_number_of_clusters(clustered)
        exp = Experiment(method=distance, command_sent="",
                         time_taken=0, k_means_plusplus=True, file_name=base.name,
                         number_of_classes=num_classes, number_of_clusters=number_of_clusters,
                         start_time=datetime.now(), comments="", f_score=float(measure))
        exp_set.experiments.append(exp)
    session.commit()


if __name__ == '__main__':
    main()
