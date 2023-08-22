import os
import yaml

def modify_deep_key(data, key, new_value):
    if key in data:
        data[key] = new_value
    for k, v in data.items():
        if isinstance(v, dict):
            modify_deep_key(v, key, new_value)

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
                        modify_deep_key(yaml_content, key, new_value)
                        with open(filepath, 'w') as outfile:
                            yaml.safe_dump(yaml_content, outfile)
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    pass
    #new_value = "/data/user/jvara/egenerator_tutorial/repositories/event-generator/gcd_preprocessing/geometry_pickles/LOM16.pickle" 
    #target_directory = "/data/user/jvara/egenerator_tutorial/repositories/event-generator/gcd_preprocessing/geometry_pickles/"  # Replace with your directory path
    #modify_yaml_key(target_directory, 'optical_module_key', 'new_value')  # Replace 'new_value' with the desired value
