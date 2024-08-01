import json

def update_accuracy_in_config(accuracy, model_name):
    # Read the existing config.json file
    config_path = '../config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update the config with the new accuracy
    config['accuracies'][model_name] = round(accuracy, 3)

    # Save the updated config back to the file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)