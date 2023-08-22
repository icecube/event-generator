import yaml

my_dict = {}
my_dict['infile_directory'] = "/data/user/jvara/simulations/lom_sim/iceprod/local_sim_test"
my_dict['model_name'] = 'lom_mcpe_june_2023'
my_dict['outfile'] = '/data/user/jvara/egenerator_tutorial/repositories/event-generator/simulation/lom_sim.i3.zst'

file_path = "./simulation_config.yaml"

with open(file_path,'w') as file:
    yaml.dump(my_dict,file)