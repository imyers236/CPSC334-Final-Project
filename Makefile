build:
	# nothing to do code in python

run: trees.py
	python3 trees.py

clean: 
	echo "Cleaning tmp directory files"
	rm -rf tmp
	rm -f tree.deb

test: test_myclassifiers.py
	pytest test_myclassifiers.py

build-deb: debBuild.sh
	./debBuild.sh

install-deb:
	sudo dpkg -i tree.deb

lint-deb: debLint.sh
	-./debLint.sh

docker-build: 
	docker build -t tree:latest .

docker-image: 
	docker run --rm tree:latest
