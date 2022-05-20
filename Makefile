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

<<<<<<< HEAD

=======
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee
convert_cnn_net:
	@python -m simple_net_cnn convert
