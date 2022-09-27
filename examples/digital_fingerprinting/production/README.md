# "Production" Digital Fingerprinting Pipeline

### Build the Morpheus container

This is necessary to get the latest changes needed for DFP

```bash
./docker/build_container_release.sh
```

### Running locally via `docker-compose`

```bash
cd examples/digital_fingerprinting/production

docker-compose build

docker-compose up
```
