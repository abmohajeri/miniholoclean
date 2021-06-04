from dataset import Dataset
from dcparser import Parser
from domain import DomainEngine
from detect import DetectEngine
from repair import RepairEngine
from evaluate import EvalEngine

# logging and logging settings
import logging
logging.basicConfig(format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
gensim_logger = logging.getLogger('gensim')
gensim_logger.setLevel(logging.WARNING)

class HoloClean:
    def __init__(self, **kwargs):
        arg_defaults = {}

        for key in kwargs:
            arg_defaults[key] = kwargs[key]

        self.session = Session(arg_defaults)

class Session:
    def __init__(self, env, name="session"):
        logging.info('initiating session with parameters: %s', env)

        # Initialize members
        self.name = name
        self.env = env
        self.ds = Dataset(name, env)
        self.dc_parser = Parser(env, self.ds)
        self.domain_engine = DomainEngine(env, self.ds)
        self.detect_engine = DetectEngine(env, self.ds)
        self.repair_engine = RepairEngine(env, self.ds)
        self.eval_engine = EvalEngine(env, self.ds)

    def load_data(self, name, fpath, na_values=None, entity_col=None, src_col=None):
        status, load_time = self.ds.load_data(name,
                                              fpath,
                                              na_values=na_values,
                                              entity_col=entity_col,
                                              src_col=src_col)
        logging.info(status)
        logging.debug('Time to load dataset: %.2f secs', load_time)

    def load_dcs(self, fpath):
        status, load_time = self.dc_parser.load_denial_constraints(fpath)
        logging.info(status)
        logging.debug('Time to load dirty data: %.2f secs', load_time)

    def get_dcs(self):
        return self.dc_parser.get_dcs()

    def detect_errors(self, detect_list):
        status, detect_time = self.detect_engine.detect_errors(detect_list)
        logging.info(status)
        logging.debug('Time to detect errors: %.2f secs', detect_time)


    def setup_domain(self):
        status, domain_time = self.domain_engine.setup()
        logging.info(status)
        logging.debug('Time to setup the domain: %.2f secs', domain_time)

    def repair_errors(self, featurizers):
        status, feat_time = self.repair_engine.setup_featurized_ds(featurizers)
        logging.info(status)
        logging.debug('Time to featurize data: %.2f secs', feat_time)
        status, setup_time = self.repair_engine.setup_repair_model()
        logging.info(status)
        logging.debug('Time to setup repair model: %.2f secs', feat_time)
        status, fit_time = self.repair_engine.fit_repair_model()
        logging.info(status)
        logging.debug('Time to fit repair model: %.2f secs', fit_time)
        status, infer_time = self.repair_engine.infer_repairs()
        logging.info(status)
        logging.debug('Time to infer correct cell values: %.2f secs', infer_time)
        status, time = self.ds.get_inferred_values()
        logging.info(status)
        logging.debug('Time to collect inferred values: %.2f secs', time)
        status, time = self.ds.get_repaired_dataset()
        logging.info(status)
        logging.debug('Time to store repaired dataset: %.2f secs', time)
        if self.env['print_fw']:
            status, time = self.repair_engine.get_featurizer_weights()
            logging.info(status)
            logging.debug('Time to store featurizer weights: %.2f secs', time)
            return status

    def evaluate(self, fpath, tid_col, attr_col, val_col, na_values=None):
        """
        evaluate generates an evaluation report with metrics (e.g. precision,
        recall) given a test set.

        :param fpath: (str) filepath to test set (ground truth) CSV file.
        :param tid_col: (str) column in CSV that corresponds to the TID.
        :param attr_col: (str) column in CSV that corresponds to the attribute.
        :param val_col: (str) column in CSV that corresponds to correct value
            for the current TID and attribute (i.e. cell).
        :param na_values: (Any) how na_values are represented in the data.

        Returns an EvalReport named tuple containing the experiment results.
        """
        name = self.ds.raw_data.name + '_clean'
        status, load_time = self.eval_engine.load_data(name, fpath, tid_col, attr_col, val_col, na_values=na_values)
        logging.info(status)
        logging.debug('Time to evaluate repairs: %.2f secs', load_time)
        status, report_time, eval_report = self.eval_engine.eval_report()
        logging.info(status)
        logging.debug('Time to generate report: %.2f secs', report_time)
        return eval_report