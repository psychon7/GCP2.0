# Data Infrastructure Makefile

.PHONY: up down restart status logs ps clean test backup restore help

# Default target
help:
	@echo "GCP Data Infrastructure Management"
	@echo ""
	@echo "Usage:"
	@echo "  make up        - Start all services"
	@echo "  make down      - Stop all services"
	@echo "  make restart   - Restart all services"
	@echo "  make status    - Show service status"
	@echo "  make logs      - Show logs from all services"
	@echo "  make ps        - List running containers"
	@echo "  make clean     - Remove all containers, volumes, and networks"
	@echo "  make test      - Run connection tests"
	@echo "  make backup    - Backup all databases"
	@echo "  make restore   - Restore databases (interactive)"
	@echo ""

# Start all services
up:
	@echo "Starting data infrastructure services..."
	docker-compose up -d
	@echo "Services started. Run 'make status' to check status."

# Stop all services
down:
	@echo "Stopping data infrastructure services..."
	docker-compose down
	@echo "Services stopped."

# Restart all services
restart:
	@echo "Restarting data infrastructure services..."
	docker-compose restart
	@echo "Services restarted."

# Show service status
status:
	@echo "Data infrastructure service status:"
	docker-compose ps

# Show logs from all services
logs:
	docker-compose logs -f

# List running containers
ps:
	docker-compose ps

# Remove all containers, volumes, and networks
clean:
	@echo "WARNING: This will remove all containers, volumes, and networks."
	@echo "This will result in DATA LOSS."
	@read -p "Are you sure you want to continue? (y/n): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		echo "Removing all containers, volumes, and networks..."; \
		docker-compose down -v; \
		echo "Clean complete."; \
	else \
		echo "Clean cancelled."; \
	fi

# Run connection tests
test:
	@echo "Running connection tests..."
	python3 test_connections.py

# Backup all databases
backup:
	@echo "Backing up PostgreSQL..."
	./scripts/backup_postgres.sh
	@echo "Backing up TimescaleDB..."
	./scripts/backup_timescaledb.sh
	@echo "Backup complete."

# Restore databases (interactive)
restore:
	@echo "Database Restore"
	@echo "1. Restore PostgreSQL"
	@echo "2. Restore TimescaleDB"
	@echo "3. Cancel"
	@read -p "Select an option (1-3): " option; \
	case $$option in \
		1) \
			read -p "Enter PostgreSQL backup file path: " file; \
			./scripts/restore_postgres.sh $$file; \
			;; \
		2) \
			read -p "Enter TimescaleDB backup file path: " file; \
			./scripts/restore_timescaledb.sh $$file; \
			;; \
		3) \
			echo "Restore cancelled."; \
			;; \
		*) \
			echo "Invalid option."; \
			;; \
	esac
