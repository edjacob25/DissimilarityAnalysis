import json
import math
import os
from configparser import ConfigParser
from datetime import datetime, timedelta
from pathlib import Path
from shutil import rmtree
from subprocess import CompletedProcess
from zipfile import ZipFile, ZIP_BZIP2
from time import sleep

import signal
import psutil
import requests
from dateutil.relativedelta import relativedelta

config = None


def get_config(section: str, config_name: str) -> str:
    global config
    if config is None:
        config = ConfigParser()
        config.read("config.ini")
    return config[section][config_name]


def format_seconds(seconds: float) -> str:
    seconds = math.fabs(seconds)
    if seconds > 3600:
        return f"{seconds / 3600} hours"
    elif seconds > 60:
        return f"{seconds / 60} minutes"
    else:
        return f"{seconds} seconds"


def format_time_lapse(start: float, end: float) -> str:
    attrs = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
    delta = relativedelta(datetime.fromtimestamp(end), datetime.fromtimestamp(start))
    spaces = ['%d %s' % (getattr(delta, attr), getattr(delta, attr) > 1 and attr or attr[:-1]) for attr in attrs if
              getattr(delta, attr)]
    return ", ".join(spaces)


def send_notification(message: str, title: str):
    data = {"body": message, "title": title, "type": "note"}
    headers = {"Content-Type": "application/json", "Access-Token": get_config("SECRETS", "pushbullet_token")}
    requests.post("https://api.pushbullet.com/v2/pushes", headers=headers, data=json.dumps(data))


def clean_auto_weka(folder: str = get_config("ROUTES", "temp_files_path")):
    path = Path(folder)
    for item in path.iterdir():
        try:
            if item.is_file():
                item.unlink()
            else:
                last_modified_timestamp = item.stat().st_mtime
                last_modified = datetime.fromtimestamp(last_modified_timestamp)
                now = datetime.now()
                diff = now - last_modified
                if diff > timedelta(minutes=20):
                    rmtree(item.resolve(), ignore_errors=True)
        except PermissionError:
            pass
        except FileNotFoundError:
            pass


def clean_temp(seconds: int = 20):
    while True:
        try:
            clean_auto_weka()
            sleep(seconds)
        except KeyboardInterrupt:
            print("Exciting cleaning")
            break


def clean_experiments(directory: Path):
    for item in directory.iterdir():
        if item.is_dir():
            rmtree(item.resolve())
            continue
        if "clustered" in item.name or "log" in item.name:
            item.unlink()


def save_results(base_directory: Path, filename: str):
    with ZipFile(base_directory / filename, 'w', compression=ZIP_BZIP2) as zipfile:
        for directory in base_directory.iterdir():
            if directory.is_dir():
                zipfile.write(directory.resolve(), arcname=directory.name)
                for file in directory.iterdir():
                    zipfile.write(file.resolve(), arcname=file.relative_to(base_directory))


def write_results(path: Path, result: CompletedProcess):
    name = path.stem
    err_path = path.with_name(f"{name}_err.log")
    log_path = path.with_name(f"{name}.log")

    with err_path.open("w") as err_file:
        err_file.write(result.stderr.decode("utf-8"))
    with log_path.open("w") as log_file:
        log_file.write(result.stdout.decode("utf-8"))


def get_platform_separator() -> str:
    if os.name is not "posix":
        return ";"
    return ":"


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def kill_java_procs():
    for proc in psutil.process_iter():
        if "java" in proc.name():
            proc.kill()
