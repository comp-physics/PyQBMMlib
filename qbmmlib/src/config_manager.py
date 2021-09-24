import yaml


class config_manager:
    def __init__(self, config_file):

        iret = self.load_config(config_file)
        return

    def load_config(self, config_file):

        print("config_mgr: Loading config from %s" % config_file)
        with open(config_file) as cf:
            self.config = yaml.load(cf, Loader=yaml.Loader)

        return

    def get_config(self):

        return self.config
