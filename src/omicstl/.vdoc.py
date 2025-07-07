# type: ignore
# flake8: noqa
#
#
#
#
#
#
import omicsTL.simulation_utils.data_utils


#
#
#
dataset_manager = DatasetManager("../data/simulated_data/large2/")
dataset_manager.scan_directory()
data_ids = dataset_manager.get_available_ids()
dataset_container = dataset_manager.load_dataset_container(data_ids[0])
#
#
#
dataset_container.split_source_data()
dataset_container.source_train_data
#
#
#
