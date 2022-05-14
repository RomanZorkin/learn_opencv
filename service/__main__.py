import logging

from service import app, app_torch

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    #app.recognize()
    app_torch.run()
