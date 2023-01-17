#!/usr/bin/make

include .env

define SERVERS_JSON
{
	"Servers": {
		"1": {
			"Name": "fastapi-alembic",
			"Group": "Servers",
			"Host": "$(DATABASE_HOST)",
			"Port": 5432,
			"MaintenanceDB": "postgres",
			"Username": "$(DATABASE_PASSWORD)",
			"SSLMode": "prefer",
			"PassFile": "/tmp/pgpassfile"
		}
	}
}
endef
export SERVERS_JSON

help:
	@echo "make"
	@echo "    install"
	@echo "        Install all packages of poetry project locally."
	@echo "    run-dev-build"
	@echo "        Run development docker compose and force build containers."
	@echo "    run-dev"
	@echo "        Run development docker compose."
	@echo "    stop-dev"
	@echo "        Stop development docker compose."
	@echo "    init-db"
	@echo "        Init database with sample data."	
	@echo "    add-dev-migration"
	@echo "        Add new database migration using alembic."
	@echo "    run-pgadmin"
	@echo "        Run pgadmin4."	
	@echo "    load-server-pgadmin"
	@echo "        Load server on pgadmin4."
	@echo "    clean-pgadmin"
	@echo "        Clean pgadmin4 data."
	@echo "    formatter"
	@echo "        Apply black formatting to code."
	@echo "    lint"
	@echo "        Lint code with ruff, and check if black formatter should be applied."
	@echo "    lint-watch"
	@echo "        Lint code with ruff in watch mode."
	@echo "    lint-fix"
	@echo "        Lint code with ruff and try to fix."	
	@echo "    run-sonarqube"
	@echo "        Starts Sonarqube container."
	@echo "    run-sonar-scanner"
	@echo "        Starts Sonarqube container."	
	@echo "    stop-sonarqube"
	@echo "        Stops Sonarqube container."

install:
	poetry shell && \
	poetry install

run-dev-build:
	docker compose -f docker-compose-dev.yml up --build

run-dev:
	docker compose -f docker-compose-dev.yml up

stop-dev:
	docker compose -f docker-compose-dev.yml down

init-db:
	docker compose -f docker-compose-dev.yml exec fastapi_server python app/initial_data.py

formatter:
	poetry run black app

run-sonarqube:
	docker compose -f docker-compose-sonarqube.yml up

run-sonar-scanner:
	docker run --rm -v "${PWD}/app:/usr/src" sonarsource/sonar-scanner-cli

stop-sonarqube:
	docker compose -f docker-compose-sonarqube.yml down

add-dev-migration:
	docker compose -f docker-compose-dev.yml exec fastapi_server alembic revision --autogenerate && \
	docker compose -f docker-compose-dev.yml exec fastapi_server alembic upgrade head

run-pgadmin:
	echo "$$SERVERS_JSON" > ./pgadmin/servers.json && \
	docker volume create pgadmin_data && \
	docker compose -f pgadmin.yml up
	
load-server-pgadmin:
	docker exec -it pgadmin python /pgadmin4/setup.py --load-servers servers.json

clean-pgadmin:
	docker volume rm pgadmin_data