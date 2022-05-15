lint:
	@flake8 service
	@mypy service

run:
	@python -m recognizer

run_model:
	@python -m service

run_nn_net:
	@python -m simple_net

run_cnn_net:
	@python -m simple_net_cnn

test_cnn_net:
	@python -m simple_net_cnn test