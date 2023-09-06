import os
import yaml

def modify_deep_key(data, key, new_value):
    modified = False
    if key in data:
        data[key] = new_value
        modified = True
    for k, v in data.items():
        if isinstance(v, dict):
            modified |= modify_deep_key(v, key, new_value)
    return modified

def modify_yaml_key(target_directory, new_value, key="optical_module_key"):
    """This function changes the path of the pickle file which
        contains the detector information on all the .yaml files of a
        exported model. Useful when a model is exported from another cluster.

        Parameters
        ----------
        target_directory : str
            Directory of the exported model.
        new_path : str
            New pickle file.
        key : str, optional
            The key name of the config files that carries the detector info.
            Default value is "optical_module_key"
        """
    for root, dirs, files in os.walk(target_directory):
        for file in files:
            if file.endswith('.yaml'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as yaml_file:
                    try:
                        yaml_content = yaml.safe_load(yaml_file)
                        modified = modify_deep_key(yaml_content, key, new_value)
                        if modified:
                            with open(filepath, 'w') as outfile:
                                yaml.safe_dump(yaml_content, outfile)
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
if __name__ == "__main__":
    new_value = "/data/user/jvara/egenerator_tutorial/repositories/event-generator/gcd_preprocessing/geometry_pickles/LOM16.pickle" 
    target_directory = "/data/user/jvara/exported_models/event-generator/lom/lom_mcpe_memory_september/"  # Replace with your directory path
    modify_yaml_key(target_directory, 'optical_module_key', 'new_value')  # Replace 'new_value' with the desired value
