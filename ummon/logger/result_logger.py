from ummon.utils.sftp_helpers import SFTP
import os

class ExperimentLogger():

    def __init__(self, logfname = "_results.plain", splunk=False):
        self.logfname = logfname
        self.splunk = splunk

    def __call__(self, **kwargs):
        self.write_dict_to_file(**kwargs)
        if self.splunk:
            self.copy_file_to_splunk()

    def copy_file_to_splunk(self):            
        try:
            con = SFTP(host="141.37.176.200", port=22, user="clusteruser", password="ios")
            con.put(self.logfname, os.path.join("/home_ext/clusteruser/index/", self.logfname))
            con.close()
        except:
            print("Could not write to Splunk-Server.")

    def write_dict_to_file(self, blob):
        with open(self.logfname , "a") as f:
            line = []
            for k,v in blob.items(): 
                line.append("{}={}, ".format(k, v).replace("\n",""))
            f.write("".join(line)[:-2] + "\n")


class GraphLogger(ExperimentLogger):

    def __init__(self, splunk=True, **kwargs):
        super(GraphLogger, self).__init__(splunk=splunk, **kwargs)

    def __call__(self, experiment_name, batch_size, num_points, coord_dims, lrate, train_dataset, 
                        epoch, optimizer, arch_multiplier, model, train_acc=0, test_acc=0, train_loss=0, test_loss=0, early_stopper=None):
        blob = {
                "exp_name"   :      experiment_name,
                "batch_size" :      batch_size,
                "num_points" :      num_points,
                "coord_dims" :      coord_dims,
                "lrate"      :      "{:.2E}".format(lrate),
                "dataset"    :      repr(train_dataset),
                "train_loss" :      "{:.2f}".format(train_loss),
                "train_acc"  :      "{:.2f}".format(train_acc),
                "test_loss"  :      "{:.2f}".format(test_loss),
                "test_acc"   :      "{:.2f}".format(test_acc),
                "epoch"      :      epoch,
                "earlystop"  :      repr(early_stopper) if early_stopper is not None else "",
                "optimizer"  :      repr(optimizer),
                "arch_multiplier" : arch_multiplier,
                "num_params" :      sum(el.numel() for el in model.parameters())
                }
        super().__call__(**blob)


class NoveltyLogger(ExperimentLogger):

    def __init__(self, splunk=True, **kwargs):
        super(NoveltyLogger, self).__init__(splunk=splunk, **kwargs)

    def __call__(self, experiment_name, reference_dataset, anomaly_dataset, model, reference_apr=0, reference_auc=0, novelty_apr=0, novelty_auc=0):
        """
        Example usage:
            NoveltyLogger()(__file__, repr(reference_dataset), repr(anomaly_dataset), repr(model), 0 ,0, 0, 0)
            > repr(model)
            > PCA(componente=512, dim_reduction="t-sne")

            > repr(anomaly)
            > Anomaly(type="squares") # {}(type={}).format(Anomaly.__class__.__name__, "squares")
        """
        blob = {
                "exp_name"          :      experiment_name,
                "reference_dataset" :      repr(reference_dataset),
                "anomaly_dataset"   :      repr(anomaly_dataset),
                "model"             :      repr(model),
                "reference_auc"     :      "{:.2f}".format(reference_auc),
                "reference_apr"     :      "{:.2f}".format(reference_apr),
                "novelty_apr"       :      "{:.2f}".format(novelty_apr),
                "novelty_auc"       :      "{:.2f}".format(novelty_auc),
            }
        super().__call__(**blob)