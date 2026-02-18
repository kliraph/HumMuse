install:
	pip install -e .

smoke:
	python -m evaluation.smoke_epic01

smoke-api:
	python -m evaluation.smoke_epic02

run-api:
	uvicorn backend.app:app --reload
