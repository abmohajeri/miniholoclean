import pandas as pd

from .detector import Detector
from utils import NULL_REPR

class OutofDomainDetector(Detector):
    """
    An error detector that treats out of domain as errors.
    """

    def __init__(self, name='OutofDomainDetector'):
        super(OutofDomainDetector, self).__init__(name)

    def setup(self, dataset, env):
        self.ds = dataset
        self.env = env
        self.df = self.ds.get_raw_data()

    def detect_noisy_cells(self):
        """
        detect_noisy_cells returns a pandas.DataFrame containing all cells with
        out of domain values.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute with out of domain for this entity
        """

        domain = {'ProviderNumber': str.isdigit,
                  'PhoneNumber': str.isdigit,
                  'ZipCode': str.isdigit}

        attributes = self.ds.get_attributes()

        errors = []
        for attr in attributes:
            if attr in domain:
                func = domain[attr]
                tmp_df = self.df[[not func(x) for x in self.df[attr]]]['_tid_'].to_frame()
                tmp_df.insert(1, "attribute", attr)
                errors.append(tmp_df)
        errors_df = pd.concat(errors, ignore_index=True)
        return errors_df

