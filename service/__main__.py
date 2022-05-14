import logging

from service import app, app_torch

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    variant = 1
    if variant == 1:
        app.recognize()
    else:
        app_torch.run()
