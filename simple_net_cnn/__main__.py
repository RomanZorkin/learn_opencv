import logging

import typer

from simple_net_cnn import app

logging.basicConfig(level=logging.DEBUG)

typer_app = typer.Typer()


@typer_app.command()
def train():
    app.run()


@typer_app.command()
def test():
    app.test()


@typer_app.command()
def convert():
    app.convert()


if __name__ == '__main__':
    typer_app()
