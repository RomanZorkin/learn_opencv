import logging

import typer

from recognizer import app

logging.basicConfig(level=logging.DEBUG)

typer_app = typer.Typer()


@typer_app.command()
def run():
    app.run()


if __name__ == '__main__':
    typer_app()
