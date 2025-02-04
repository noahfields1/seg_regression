import os
# os.environ['CUDA_VISIBILE_DEVICES'] = ''
import modules.io as io
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('config_file')
parser.add_argument('data_key')

args = parser.parse_args()

config   = io.load_yaml(args.config_file)
DATA_KEY = args.data_key

#dataset
import factories.dataset_factory as dataset_factory
data    = dataset_factory.get(config, DATA_KEY)

#model
import factories.model_factory as model_factory

if os.path.isdir(config['MODEL_DIR']):
    iter = int(open(config['ITER_FILE']).readlines()[0])
    model = model_factory.get(config)
    model.load()
    model.global_step += iter
    model.start_iter  = iter
    print("loaded model, restarting from iteration {}".format(iter))
else:
    model = model_factory.get(config)
    print("no model found, training fresh")

#preprocessor
import factories.preprocessor_factory as prepro_factory
preprocessor = prepro_factory.get(config)

#trainer
import factories.trainer_factory as trainer_factory
trainer = trainer_factory.get(config)
trainer.setup_directories()

trainer.set_data(data, DATA_KEY)
trainer.set_preprocessor(preprocessor)

trainer.set_model(model)

trainer.train()

trainer.save()
