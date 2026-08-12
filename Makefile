.PHONY: test test-bag run down

test:
	PYTHONPATH=src pytest

test-bag:
	@test -n "$(BAG)" || (echo "usage: make test-bag BAG=/path/to/bag ARGS='--service lidar-voxel --expect lidar.proximity'"; exit 2)
	PYTHONPATH=src python3 -m robotkit.testing.bag_replay "$(BAG)" $(ARGS)

run:
	docker compose up --build

down:
	docker compose down
