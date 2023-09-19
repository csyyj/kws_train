import logging.config

LOGGING = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s [%(levelname)-8s]: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        } 
    },
    'filters': {

    },
    'handlers': {
        'file': {
            'level': 'INFO', 
            'class': 'logging.FileHandler',
            'filename': 'train.log'
        },
        'console': {
            'level': 'INFO',
            'formatter': 'default',
            'class': 'logging.StreamHandler'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

logging.config.dictConfig(LOGGING)
