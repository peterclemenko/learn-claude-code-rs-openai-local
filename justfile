# Default task: list recipes
default:
    @just --list

# Toolchain setup / override check
setup:
    rustup show

# Build all workspace packages
build:
    cargo build --workspace

# Run the application
run:
    cargo run

# Run the full workspace test suite
test:
    cargo test --workspace

# Run tests with all features enabled
test-all:
    cargo test --workspace --all-features

# Run clippy checks across all packages and targets
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Format all Rust crates in the workspace
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Check compilation across the workspace
check:
    cargo check --workspace

# Verify TOON serialization and token-efficiency functionality
test-toon:
    cargo test --workspace toon

# Verify OpenTelemetry instrumentation
test-otel:
    cargo test --workspace otel

# Run optional local-inference integration tests
test-local:
    cargo test --workspace --features local-integration -- --ignored

# Complete local CI verification
ci: fmt-check lint test