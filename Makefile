.PHONY: twine twine_test

twine: build upload

twine_test: build upload_test

build:
	pdm run python -m build

upload:
	pdm run twine upload --repository pypi dist/*

upload_test:
	pdm run twine upload --repository testpypi dist/*
