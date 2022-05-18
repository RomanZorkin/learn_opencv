lint:
	@flake8 service
	@mypy service

recognize:
	@python -m recognizer

run_model:
	@python -m service

run_nn_net:
	@python -m simple_net

train_cnn_net:
	@python -m simple_net_cnn train

test_cnn_net:
	@python -m simple_net_cnn test


convert_cnn_net:
	@python -m simple_net_cnn convert
