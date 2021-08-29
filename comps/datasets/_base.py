"""
Base I/O for loading datsets.
"""

import pyarrow as pa
from pathlib import Path 
from pyarrow import csv
from sklearn.utils import Bunch

from mystique.utils.tools import root_path, import_text


def load_dataset(
    dsname, full=False, frame=None, id_names=None, target_names=None,
    feature_names=None):
    """Load dataset and return dataset data and metadata as dict like object.

    Parameters
    ----------
    dsname : str
        Name of dataset load.
    frame : bool
        Add frame attribute padas.DataFrame representation of dataset. 
    full : bool
        Return full or larger version of dataset if available

    Returns
    -------
    dataset : dictlike object
        Dictionary like sklearn.utils.Bunch object with different forms of
        dataset data and metadata for test and practice use.
    """ 
    dataset_root = root_path().joinpath("datasets", "data", dsname) 
    dataset_fname = "{0}.csv.gz".format(dsname) if not full else "{0}_full.csv.gz".format(dsname)

    dataset_descr = import_text(dataset_root.joinpath("{0}_descr.rst".format(dsname)))
    pyarrow_table = csv.read_csv(dataset_root.joinpath(dataset_fname))

    frame = pyarrow_table.as_pandas() if frame else None

    return Bunch (
        table=pyarrow_table,
        data=pyarrow_table.to_pandas().to_numpy(),
        columns=pyarrow_table.column_names,
        frame=frame,
        DESCR=dataset_descr,
        feature_names=feature_names,
        target_names=target_names,
        id_names=id_names,
    )


