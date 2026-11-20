# Default task: list recipes
default:
    @just --list

# Toolchain setup / override check
setup:
    rustup show

# Build all workspace packages
build:
    nix develop --command cargo build --workspace

# Run the application
run:
    nix develop --command bash -c 'cargo build --workspace && exec env SSL_CERT_FILE="$SSL_CERT_FILE" SYSTEM_CERT_FILE="$SYSTEM_CERT_FILE" SSL_CERT_DIR="$SSL_CERT_DIR" ./target/debug/learn-claude-code-rs-openai-local'

# Run the full workspace test suite
test:
    RUST_TEST_THREADS=1 nix develop --command cargo test --workspace -- --nocapture

# Run tests with all features enabled
test-all:
    RUST_TEST_THREADS=1 OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}" nix develop --command cargo test --workspace --all-features -- --nocapture

# Run clippy checks across all packages and targets
lint:
    nix develop --command cargo clippy --workspace --all-targets --all-features -- -D warnings

# Format all Rust crates in the workspace
fmt:
    nix develop --command cargo fmt --all

# Check formatting without modifying files
fmt-check:
    nix develop --command cargo fmt --all -- --check

# Check compilation across the workspace
check:
    nix develop --command cargo check --workspace

# Verify TOON serialization and token-efficiency functionality
test-toon:
    nix develop --command cargo test --workspace toon

# Verify OpenTelemetry instrumentation
test-otel:
    RUST_TEST_THREADS=1 nix develop --command cargo test --workspace otel -- --nocapture

# Run optional local-inference integration tests
test-local:
    nix develop --command cargo test --workspace --features local-integration -- --ignored

# Complete local CI verification
ci: fmt-check lint test