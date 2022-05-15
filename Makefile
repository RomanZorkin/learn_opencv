lint:
	@flake8 service
	@mypy service

run_model:
	@python -m service

run_nn_net:
	@python -m simple_net

run_cnn_net:
	@python -m simple_net_cnn
