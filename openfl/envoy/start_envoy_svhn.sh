#!/bin/bash
set -e

fx envoy start -n env_two --disable-tls --envoy-config-path envoy_config_svhn.yaml -dh director -dp 50051