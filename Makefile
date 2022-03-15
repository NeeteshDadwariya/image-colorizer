env-update:
	conda activate cuda
	conda env update --file environment.yml --prune
