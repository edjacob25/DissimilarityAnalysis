from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


@dataclass
class GeneralParameters:
    directory: Path
    verbose: bool
    classpath: str = None
    measure_calculator_path: str = None
    normalize_dissimilarity: bool = False


@dataclass
class ExperimentSetParameters:
    initial: bool = True
    alternate: bool = False
    strategy: str = None
    multiplier: str = None
    weight: str = None
    description: str = ""


@dataclass
class ExperimentParameters:
    filepath: Path
    measure: str
    strategy: str = None
    weight_strategy: str = None
    learning_based: bool = False
    auc_type: str = None


class Experiment(Base):
    __tablename__ = "experiment"

    id = Column(Integer, primary_key=True)
    method = Column(String)
    f_score = Column(Float)
    command_sent = Column(String)
    time_taken = Column(Float)
    k_means_plusplus = Column(Boolean)
    file_name = Column(String)
    comments = Column(String)
    number_of_classes = Column(Integer)
    number_of_clusters = Column(Integer)
    start_time = Column(DateTime)

    set_id = Column(Integer, ForeignKey("experiment_set.id"))
    set = relationship("ExperimentSet", back_populates="experiments")


class ExperimentSet(Base):
    __tablename__ = "experiment_set"

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    time_taken = Column(Float)
    number_of_datasets = Column(Integer)
    base_directory = Column(String)
    commit = Column(String)
    description = Column(String)
    experiments = relationship("Experiment", order_by=Experiment.id, back_populates="set")
