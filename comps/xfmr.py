"""
Module for easier transformation and tracking between different data structures
when using pandas, scikit-learn, NumPy, SciPy, and Vaex.
"""
from collections import deque
from vaex import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin



class PandasTypeSelector(BaseEstimator, TransformerMixin):
    """For transforming and encoding Pandas DataFrame columns in sklearn pipelines."""
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Returns list of columns with a specific data type.

        Parameters
        ----------
        X : DataFrame
            pandas.DataFrame to return subset of.

        Returns
        -------
        DataFrame
            The column subset of the X DataFrame with dtype specified when the
            ColumnTypeSelector class was initialized.
        """
        assert isinstance(X, pd.DataFrame), "Target is not a Pandas DataFrame"
        return X.select_dtypes(include=[self.dtype])

class Xfmr:
    """Core class used as base to store and transform data.

    The Xfmr is the primary data container for the transformer package. It
    provides a standardized interface for storing and managing data using
    various types of data structures and supports the creation of make domain
    specific relationships between the different data structure intances or
    elements.  Additionally, it provides a metadata library that can be used to
    add additional metadata about the data structures and relationships encoded
    into the container elements. 

    Parameters
    ----------
    t : PyArrow table or array of PyArrow tables
        PyArrow tables are stored in the table attribute by name and can
        accessed via slicing.

    s : xfmr.Stack class instance or list of instances


    Attributes
    ----------
    arrays : dict of pyarrow.array class instance or list of instances. 
        TBD

    tables : PyArrow table or dict of PyArrow tables
        PyArrow tables are stored in the table attribute by name and can
        accessed via slicing.
    
    deques: dict of collections.deque class instances. 
        TBD

    names : str or list of str
        Names of all the data structures in the transformer.
    
    mdata : xfmr.Metadata class instance
        TBD
    """
    def __init__(self, **kwargs):
        self.arrays = None
        self.deques = None
        self.frames = None
        self.metaid = None
        self.meta = None

    


