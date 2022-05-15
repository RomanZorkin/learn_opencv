import logging

from simple_net_cnn import app

logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    variant = 1
    if variant == 1:
        app.run()
    else:
        app.test()
