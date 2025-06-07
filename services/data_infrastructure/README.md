# Data Infrastructure Service

This service provides the database and messaging infrastructure for the Global Consciousness Project (GCP). It includes:

- PostgreSQL 16 with TimescaleDB 2.15 extension for time-series data
- PostgreSQL 16 for relational data
- Redis 7 for caching
- NATS JetStream 3.x for messaging

## Components

### PostgreSQL with TimescaleDB

TimescaleDB is used for storing time-series data such as:
- Entropy metrics
- Coherence analytics
- Health monitoring data

### PostgreSQL (Standard)

Standard PostgreSQL is used for storing relational data such as:
- Camera registry
- Experiment configurations
- User data

### Redis

Redis is used for:
- Caching frequently accessed data
- Session management
- Rate limiting
- Temporary storage for event correlation results

### NATS JetStream

NATS JetStream is used for:
- Reliable messaging between services
- Event streaming
- Persistent message storage

## Setup

### Local Development

To start the infrastructure locally:

```bash
docker-compose up -d
```

This will start all components with the configuration defined in the docker-compose.yml file.

### Configuration

Configuration for each component is stored in the `config` directory:
- `postgres/`: PostgreSQL configuration files
- `timescaledb/`: TimescaleDB specific configuration
- `redis/`: Redis configuration files
- `nats/`: NATS configuration files

### Initialization

Database initialization scripts are stored in the `init` directory:
- `postgres/`: PostgreSQL initialization scripts
- `timescaledb/`: TimescaleDB specific initialization scripts

## Usage

### Connection Information

#### PostgreSQL

- Host: localhost
- Port: 5432
- Database: gcp
- Username: gcp_user
- Password: See .env file

#### TimescaleDB

- Host: localhost
- Port: 5433
- Database: gcp_timeseries
- Username: gcp_user
- Password: See .env file

#### Redis

- Host: localhost
- Port: 6379
- Password: See .env file

#### NATS

- Host: localhost
- Port: 4222
- Username: gcp_user
- Password: See .env file

### Health Checks

Health check endpoints:
- PostgreSQL: `localhost:5432/health`
- TimescaleDB: `localhost:5433/health`
- Redis: `localhost:6379/health`
- NATS: `localhost:4222/healthz`

## Monitoring

Prometheus exporters are configured for each component:
- PostgreSQL: `localhost:9187`
- Redis: `localhost:9121`
- NATS: `localhost:7777`

## Backup and Recovery

Backup scripts are provided in the `scripts` directory:
- `backup_postgres.sh`: Backup PostgreSQL databases
- `backup_timescaledb.sh`: Backup TimescaleDB databases
- `restore_postgres.sh`: Restore PostgreSQL databases
- `restore_timescaledb.sh`: Restore TimescaleDB databases

## Troubleshooting

Common issues and their solutions:
- Connection refused: Check if the service is running and the port is correct
- Authentication failed: Check the username and password in the .env file
- Database not found: Check if the database has been created

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/16/index.html)
- [TimescaleDB Documentation](https://docs.timescale.com/latest/main)
- [Redis Documentation](https://redis.io/documentation)
- [NATS Documentation](https://docs.nats.io/)
