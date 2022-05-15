import logging

from simple_net import app

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    run_opt = 2
    if run_opt == 1:
        app.simple_gradient()
    elif run_opt == 2:
        app.create_nn(200, 0.01, 10, 10)
