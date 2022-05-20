import logging

import typer

from recognizer import app

logging.basicConfig(level=logging.INFO)

typer_app = typer.Typer()


@typer_app.command()
def run():
    app.recognize()


if __name__ == '__main__':
    typer_app()
