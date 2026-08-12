.PHONY: test run down

test:
	PYTHONPATH=src pytest

run:
	docker compose up --build

down:
	docker compose down
