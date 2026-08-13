.PHONY: test test-bag run

test:
	PYTHONPATH=src pytest

test-bag:
	@test -n "$(BAG)" || (echo "usage: make test-bag BAG=/path/to/bag ARGS='--service lidar-voxel --expect lidar.proximity'"; exit 2)
	PYTHONPATH=src python3 -m robotkit.testing.bag_replay "$(BAG)" $(ARGS)

run:
	wendy run
