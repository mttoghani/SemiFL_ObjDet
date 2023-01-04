import yaml

global cfg
if 'cfg' not in globals():
    with open('config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


def process_args(args):
    for k in cfg:
        cfg[k] = args[k]
    if args['control_name']:
        control_name_list = args['control_name'].split('_')
        control_keys_list = list(cfg['control'].keys())
        cfg['control'] = {control_keys_list[i]: control_name_list[i] for i in range(len(control_name_list))}
    cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
    return
