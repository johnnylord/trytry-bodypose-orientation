import os
import os.path as osp
import argparse
import yaml
from agent import get_agent_cls


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True, help="path to configuration file")

def main(config_path):
    with open(config_path) as f:
        config = yaml.full_load(f)
    agent_cls = get_agent_cls(config['agent'])
    agent = agent_cls(config)
    agent.train()
    agent.finalize()

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args['config'])
