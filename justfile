# Default task: list recipes
default:
    @just --list

# Toolchain setup / override check
setup:
    rustup show

# Build all workspace packages
build:
    cargo build --workspace

# Run a specific chapter package (e.g., just run s01_agent_loop)
run package:
    cargo run -p {{package}}

# Run tests with OTel subscriber active
test:
    cargo test --workspace

# Run clippy checks across all packages
lint:
    cargo clippy --workspace --all-targets -- -D warnings


# Format all Rust crates in the workspace
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check