"""
Utility functions helpful for general usage
"""
import os
import re
import sys
import csv
import yaml
import pkg_resources
from pathlib import Path
from getpass import getuser
from uuid import uuid4
from datetime import datetime
from collections import namedtuple
from subprocess import run
from functools import wraps, reduce
from collections.abc import Iterable


def root_path():
    """Returns absolute root path for package as Path object"""
    return Path(pkg_resources.resource_filename(__name__, "")).parent


def check_path(path, create=False):
    """Check if path to directory exists and optional create if not exists"""
    path_exists = os.path.exists(path)
    if create and not path_exists:
        os.makedirs(path)
        return True
    else:
        return path_exists


def user_path():
    """Returns path to user's local home directory Windows and Unix"""
    platform, user = sys.platform, getuser()
    if bool(re.match(r"win.*", platform)):
        drive = os.path.splitdrive(sys.executable)[0]
        user_path = Path(drive).joinpath("Users", user)
        user_path = user_path if check_path(user_path) else None
    else:
        user_path = Path(os.getenv("HOME"))
        user_path = user_path if check_path(user_path) else None
    return user_path


def check_file(fpath):
    """Check that that path leads to valid file"""
    if fpath is None:
        raise FileNotFoundError("'None' is not a file path")
    elif not Path(fpath).is_file():
        raise FileNotFoundError("Path '{0}' does not yield valid file".format(fpath))
    else:
        return True


def append_new_line(fpath, text):
    """Append given text as a new line at the end of file"""
    with open(fpath, "a+") as openfile:
        openfile.seek(0)
        data = openfile.read(100)
        if len(data) > 0:
            openfile.write("\n")
        openfile.write(text)
        

def import_text(fpath, return_type="string"):
    """Returns single string representation of text file or list of text lines"""
    file = Path(fpath)
    check_file(file)
    with open(file) as openfile:
        text = openfile.readlines() if return_type == "lines" else openfile.read()
    return text


def import_csv_dict(csv_fpath):
    """Returns a key: value dictionary from two column CSV"""
    with open(csv_fpath) as openfile:
        csv_dict = dict(filter(None, csv.reader(openfile)))
    return csv_dict


def import_csv_list(csv_fpath):
    """Returns a list of dictionaries where header CSV row is keys for each dict
    and values in each dict are corresponding column value of rows > 1
    """
    with open(csv_fpath) as openfile:
        dicts_list = list(csv.DictReader(openfile))
    return dicts_list


def import_yaml(fpath):
    """Returns a dictionary with imported YAML file data"""
    file = Path(fpath)
    check_file(file)
    file_name, file_extension = os.path.splitext(file)
    assert file_extension in {".yaml", "yml"}, "YAML file must have .yaml or .yml extension"
    with open(file) as openfile:
        yaml_dict = yaml.load(openfile, Loader=yaml.Loader)
    return yaml_dict


def list_filter_match(iterable, repattern):
    """Take an iterable and filters with regex match and returns a list"""
    pattern = re.compile(repattern)
    return list(filter(lambda x: bool(pattern.match(x)), iterable))   


def find_item(tgt_dict, key):
    """Recursively search dictionary for key and return value"""
    if key in tgt_dict: return tgt_dict[key]
    for k, v in tgt_dict.items():
        if isinstance(v, dict):
            item = find_item(v, key)
            if item is not None:
                return item


def replace_dict_keys(tgt_dict, map_dict):
    """
    Recursively iterates over a target dictionary and replaces existing keys
    based on map dictionary with existing keys in the target mapped to new
    values and returns a new dictionary with the keys replaced
    """
    new_dict = {}
    for key in tgt_dict.keys():
        new_key = map_dict.get(key, key)
        if isinstance(tgt_dict[key], dict):
            new_dict[new_key] = replace_dict_keys(tgt_dict[key], map_dict)
        else:
            new_dict[new_key] = tgt_dict[key]
    return new_dict


def check_args(allowed, arg):
    """Raise name error if argument doesn't match 'allowed' list/set of values"""
    assert type(allowed) in {list, set}, "First argument must be 'allowed' list/set of args values"
    if arg not in allowed:
        raise NameError("Unexpected arg {0}: allowed args are {1}".format(arg, allowed.__str__()))
    return True


def check_kwargs(allowed, **kwargs):
    """Raise error for keyword arguments not in allowed list/set else return True"""
    assert type(allowed) in {list, set}, ("First argument must be 'allowed' "
        "list/set of keyword argument names")
    for kwarg in kwargs:
        if kwarg not in allowed:
            raise NameError("Unexpected kwarg {0}: allowed kwargs are {1}".format(kwarg, allowed.__str__()))
    return True


def eventstamp(**kwargs):
    """Returns a named tuple with uniform unique ID and UTC timestamp string"""
    check_kwargs(["uuid", "utc"], **kwargs)
    uuid, utc = None, None
    if kwargs:
        uuid, utc = kwargs.get("uuid"), kwargs.get("utc")        
    uuid = str(uuid4()) if not uuid else uuid
    utc = datetime.utcnow().isoformat() if not utc else utc
    Event = namedtuple("Event", ["uuid", "utc"])
    return Event(uuid, utc)


def has_attrs(obj, attrs):
    """Check that an object has specific attributes"""
    assert type(attrs) in {list, set}, "Second argument must be list/set of attributes"
    for attr in attrs:
        if not hasattr(obj, attr):
            raise ValueError("Required metadata attribute '{0}' is missing, "
                "check data object metadata".format(attr))
        if not getattr(obj, attr):
            raise ValueError("Required metadata attribute '{0}' does not have a "
                "value, check data object metadata".format( attr))
    return True


def format_kwargs(string, *args):
    """Return set or dict (optional) of keyword arguments ({keyword}) in provided text"""
    keywords = {x.strip('{}') for x in re.findall(r'((?<!{){(?!{).*?(?<!})}(?!}))', string)}
    if args and args[0] == "dict":
        return {sub: "" for sub in keywords}
    return keywords


def import_string_format(fpath, *arg):
    """
    Create string from text file and optionally apply dictionary of string
    format **mapping substitutions.
    """
    check_file(fpath)
    string = import_text(fpath)
    if arg:
        keywords = format_kwargs(string)
        assert isinstance(arg[0], dict), ("Optional argument must be dictionary of "
            "key:value pairs where value should replaces key in string")
        subs_dict = arg[0]
        for keyword in keywords:
            if keyword not in subs_dict.keys():
                raise ValueError("{0} kwarg missing from substitutions dict".format(keyword))
        string = string.format(**subs_dict)
    return string


def ask_yesno(question):
    """
    Helper to get yes / no answer from user.
    """
    yes = {'yes', 'y'}
    no = {'no', 'n'}  # pylint: disable=invalid-name
    done = False
    print(question)
    while not done:
        choice = input().lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond by yes or no.")

 
class DictQuery(dict):
    """This class allows one to easily work with a nested dictionary.

    This class accepts a nested dictionary as an argument, then allows the get
    method to use a dot separated path of dictionary keys, and returns the
    corresponding value.

    Args:
        dict: Nested dictionary type argument.
    """
    def get(self, path, default=None):
        """Return value from path of dot separated keys"""
        keys, val = path.split("."), None
        for key in keys:
            if val:
                if isinstance(val, list):
                    val = [v.get(key, default) if v else None for v in val]
                else:
                    val = val.get(key, default)
            else:
                val = dict.get(self, key, default)
            if not val:
                break
        return val


def dict_to_list(dictionary):
    """Take dictionary and serialize into a list and filter out None values"""
    assert isinstance(dictionary, dict), "Argument must be of type dict"
    dict_items = dictionary.items()
    lst = [item for t in dict_items for item in t]
    return list(filter(None, lst))


def func_iter(func):
    """
    Decorator to apply function/method to all objects if iterable argument is
    provided as only general positional argument
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        def iter_check(arg):
            return not isinstance(arg, str) and isinstance(arg, Iterable)

        is_iter = bool(list(filter(iter_check, args)))
        if is_iter:
            iter_loc = [(i, v) for i, v in enumerate(args) if iter_check(v)][0]
            if iter_loc[0] > 0:
                func_applied = [func(args[0], x, **kwargs) for x in args[1]]
            else:
                func_applied = [func(x, **kwargs) for x in args[0]]
            return func_applied
        return func(*args, **kwargs)
    return wrapper


def print_return(func):
    """
    Prints return object and returns if function returns printable object
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        rtrn = func(*args, **kwargs)
        if rtrn is not None:
            print(rtrn)
            return rtrn[0] if isinstance(rtrn[0], tuple) else rtrn
    return wrapper