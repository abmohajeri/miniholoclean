import pandas as pd
import numpy as np

from sklearn.cluster import AffinityPropagation
from fuzzywuzzy import fuzz

from .detector import Detector

class AnomalyDetector(Detector):
    """
    An error detector that treats out of domain as errors.
    """

    def __init__(self, name='AnomalyDetector'):
        super(AnomalyDetector, self).__init__(name)

    def setup(self, dataset, env):
        self.ds = dataset
        self.env = env
        self.df = self.ds.get_raw_data()

    def detect_noisy_cells(self):
        """
        detect_noisy_cells returns a pandas.DataFrame containing all cells with
        anomaly values.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute with out of domain for this entity
        """

        attributes = self.ds.get_attributes()

        # Hospital
        constraints = [
            ['Condition', 'MeasureName', 'HospitalType'],
            ['HospitalName', 'ZipCode'],
            ['HospitalName', 'PhoneNumber'],
            ['MeasureCode', 'MeasureName'],
            ['MeasureCode', 'Stateavg'],
            ['ProviderNumber', 'HospitalName'],
            ['MeasureCode', 'Condition'],
            ['HospitalName', 'Address1'],
            ['HospitalName', 'HospitalOwner'],
            ['HospitalName', 'ProviderNumber'],
            ['HospitalName', 'PhoneNumber', 'HospitalOwner', 'State'],
            ['City', 'CountyName'],
            ['ZipCode', 'EmergencyService'],
            ['HospitalName', 'City'],
            ['MeasureName', 'MeasureCode']
        ]
        for x in attributes:
            constraints.append([x])

        errors_df = pd.DataFrame(columns=['_tid_', 'attribute'])
        for constraint in constraints:
            h = self.df[constraint] \
                .groupby(constraint) \
                .size().to_frame().rename(columns={0: "Count"}).reset_index() \
                .sort_values('Count', ascending=False)

            mean = h[['Count']].mean().iloc[0]
            words = h[constraint[0]]
            words = np.asarray(words)
            similarity = np.array([[fuzz.token_sort_ratio(w1, w2) for w1 in words] for w2 in words])
            np.fill_diagonal(similarity, 10000)
            model = AffinityPropagation(damping=0.9, random_state=0, max_iter=2000)
            model.fit(similarity)
            for cluster_id in np.unique(model.labels_):
                exemplar = words[model.cluster_centers_indices_[cluster_id]]
                cluster = words[np.nonzero(model.labels_ == cluster_id)]
                arr = np.append(cluster, [exemplar])
                for a in arr:
                    search = self.df[self.df[constraint[0]] == a]
                    cnt = h[h[constraint[0]] == a].iloc[0]['Count']
                    if cnt < mean:
                        tmp_df = search['_tid_'].to_frame()
                        tmp_df.insert(1, 'attribute', constraint[0])
                        errors_df = errors_df.append(tmp_df)

        return errors_df